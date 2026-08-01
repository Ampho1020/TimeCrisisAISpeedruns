-- Minimal comm.* bridge for BizHawk 2.11.1 <-> Python ES training.
-- Open via: Tools -> Lua Console -> Open Script
--
-- IMPORTANT LAUNCH REQUIREMENT
--   BizHawk's comm socket server must be initialized at launch, NOT from Lua.
--   Start Python (the listener) FIRST, then launch BizHawk like:
--       ./EmuHawk --socket_ip=127.0.0.1 --socket_port=8765
--   Do NOT call comm.socketServerSetIp/SetPort here -- if the server was
--   initialized via the command line those calls can null-ref / tear down
--   the connection.
--
-- STARTUP ORDER (matters!):
--   1. python es_train.py        (binds 127.0.0.1:8765, waits on accept)
--   2. launch BizHawk WITH the socket flags above
--   3. load the game (Guncon port, savestate slot 1 past in-game calibration)
--   4. Tools -> Lua Console -> open this script
--
-- HANDSHAKE + DEADLOCK NOTE
--   BizHawk's socket connects at launch, but only THIS script answers commands.
--   So Python's accept() can succeed long before the script is loaded, causing
--   the first command to time out. To avoid that race, this script sends a
--   single "READY" line as soon as it starts. bridge_client.connect() blocks
--   until it receives that line before issuing any command.
--
--   CRITICAL: comm socket I/O only pumps while the frame loop advances. If we
--   send READY and then immediately block reading a command, READY never
--   flushes -> Python waits for READY, Lua waits for a command -> deadlock, and
--   BizHawk appears frozen. So we do ONE emu.frameadvance() right after sending
--   READY to flush it out the transport before we ever read.
--
--   BLOCKING READ NOTE: on this build comm.socketServerResponse() BLOCKS until a
--   message arrives (there is no non-blocking variant -- confirmed via
--   getluafunctionslist). A blocking read would stall the frame loop forever, so
--   we call comm.socketServerSetTimeout(1) to give it a ~1ms receive timeout.
--   On timeout it returns an empty string, which the loop guard treats as "no
--   command this frame".
--
--   FRAME BUDGET NOTE: the loop advances a real game frame ONLY when a "step" is
--   pending. All other commands (reads, set_input, load/save, frame, hud) use
--   emu.yield() instead, which keeps the loop/UI alive and the transport pumping
--   WITHOUT consuming a frame. This keeps the decision cadence at exactly
--   FRAME_SKIP frames per tick and keeps throughput from being throttled to the
--   emulator's frame rate.
--
-- Commands (one per line):
--   read_u16 <addr>                               -> OK <value>
--   set_input <shoot01> <cover01> <aim_x> <aim_y> -> OK   (aim_* optional)
--   step <n>                                      -> OK
--   load <slot> / save <slot>                     -> OK
--   frame                                         -> OK <framecount>
--   hud <line1|line2|...> / hud_clear             -> OK
--
-- NOTES
--   * Aim X/Y are now written to the Guncon axes via joypad.setanalog, so the
--     AI -- not the host mouse -- controls where the gun points. aim_x/aim_y
--     arrive normalized 0..1 (0 = left/top) and are mapped to the axis ranges
--     below. Vertical is currently held centered by the caller until vision
--     lands; horizontal is driven by the policy.
--   * Trigger (shoot) and the cover/peek button are driven INDEPENDENTLY. We do
--     NOT couple them (no "shoot and not cover"): the game itself only registers
--     a shot when fully out of cover, so we just forward both button states and
--     let the AI learn the hold-to-shoot timing.
--   * Confirmed Guncon key names on this build:
--       trigger = "P1 Trigger" (left mouse / shoot)
--       cover   = "P1 A"       (right mouse)
--       axes    = "P1 X Axis" (0..2640), "P1 Y Axis" (16..256)

local DOMAIN = "MainRAM"

-- Debug input logging. Set true to get READ_BACK lines in the Lua console.
-- Must be declared here (top of file) so apply_input() and startup code both see it.
local DEBUG_INPUT_LOG = true   -- set false for normal training
local debug_frame_count = 0

local shoot, cover = false, false
local aim_x_norm, aim_y_norm = 0.5, 0.5   -- stored only; not written yet
local hud_lines = {}

-- pending frames to advance, consumed one-per-iteration by the main loop
local pending_steps = 0

-- Confirmed Guncon controls on this build.
-- Keys are WITHOUT the "P1 " prefix on this core/build -- confirmed by joypad.get() dump.
local GUNCON_TRIGGER_KEY = "Trigger"
local GUNCON_COVER_KEY   = "A"
local GUNCON_AIM_X_KEY   = "X Axis"
local GUNCON_AIM_Y_KEY   = "Y Axis"
-- Axis ranges the Guncon expects (from RAM/input probing on this build).
local GUNCON_X_MIN, GUNCON_X_MAX = 0, 2640
local GUNCON_Y_MIN, GUNCON_Y_MAX = 16, 256

local function parse_int(s)
  if not s then return nil end
  if string.sub(s, 1, 2) == "0x" or string.sub(s, 1, 2) == "0X" then
    return tonumber(s)
  end
  return tonumber(s) or tonumber("0x" .. s)
end

local function clamp01(v)
  if v < 0.0 then return 0.0 end
  if v > 1.0 then return 1.0 end
  return v
end

local function draw_hud()
  for i, txt in ipairs(hud_lines) do
    gui.text(4, 4 + (i - 1) * 14, txt, 0xFFFFFF00, 0xC0000000)
  end
end

local function apply_input()
  -- Buttons: forward trigger and cover/peek INDEPENDENTLY. The game only lets a
  -- shot register when fully out of cover, so we don't couple them here -- the
  -- AI learns the timing (and the ~0.2s cover transition) on its own.
  --
  -- IMPORTANT: cover=true means the A button IS PRESSED, which causes the
  -- character to LEAVE cover (exposed). cover=false = A released = IN cover.
  -- The Python variable name is the inverse of the in-game cover state.
  joypad.set({
    [GUNCON_TRIGGER_KEY] = shoot,
    [GUNCON_COVER_KEY]   = cover,
  })
  -- Aim: drive the Guncon axes ourselves so the AI, not the host mouse, aims.
  -- Map normalized 0..1 (0 = left/top) onto the axis ranges and round to int.
  local ax = GUNCON_X_MIN + aim_x_norm * (GUNCON_X_MAX - GUNCON_X_MIN)
  local ay = GUNCON_Y_MIN + aim_y_norm * (GUNCON_Y_MAX - GUNCON_Y_MIN)
  joypad.setanalog({
    [GUNCON_AIM_X_KEY] = math.floor(ax + 0.5),
    [GUNCON_AIM_Y_KEY] = math.floor(ay + 0.5),
  })

  -- Verify: read the input state back so we can confirm BizHawk is applying
  -- our overrides (not letting hardware/mouse bleed through).
  if DEBUG_INPUT_LOG and emu.framecount() % 30 == 0 then
    local ok2, actual = pcall(joypad.get, 1)
    local a_actual = ok2 and actual and tostring(actual[GUNCON_COVER_KEY])   or "?"
    local t_actual = ok2 and actual and tostring(actual[GUNCON_TRIGGER_KEY]) or "?"
    print(string.format(
      "[apply fc=%d] SET cover=%s shoot=%s | READ_BACK P1A=%s trigger=%s | aim=(%.3f,%.3f)",
      emu.framecount(), tostring(cover), tostring(shoot),
      a_actual, t_actual, aim_x_norm, aim_y_norm))
  end
end

local function handle(line)
  local parts = {}
  for w in string.gmatch(line, "%S+") do table.insert(parts, w) end
  local cmd = parts[1]
  if not cmd then return "ERR empty\n" end

  if cmd == "read_u16" then
    local addr = parse_int(parts[2])
    if not addr then return "ERR bad_addr\n" end
    return "OK " .. tostring(memory.read_u16_le(addr, DOMAIN)) .. "\n"

  elseif cmd == "set_input" then
    shoot = (parts[2] == "1")
    cover = (parts[3] == "1")
    if parts[4] ~= nil then aim_x_norm = clamp01(tonumber(parts[4]) or 0.5) end
    if parts[5] ~= nil then aim_y_norm = clamp01(tonumber(parts[5]) or 0.5) end
    return "OK\n"

  elseif cmd == "step" then
    -- Do NOT frameadvance here; queue it for the main loop. Advancing frames
    -- inside a comm handler can fault BizHawk 2.11.1.
    pending_steps = pending_steps + (tonumber(parts[2]) or 1)
    return "OK\n"

  elseif cmd == "load" then
    savestate.loadslot(tonumber(parts[2]) or 1)
    return "OK\n"

  elseif cmd == "save" then
    savestate.saveslot(tonumber(parts[2]) or 1)
    return "OK\n"

  elseif cmd == "frame" then
    return "OK " .. tostring(emu.framecount()) .. "\n"

  elseif cmd == "hud" then
    hud_lines = {}
    local text = string.sub(line, 5)
    for seg in string.gmatch(text, "[^|]+") do
      table.insert(hud_lines, seg)
    end
    return "OK\n"

  elseif cmd == "hud_clear" then
    hud_lines = {}
    return "OK\n"

  else
    return "ERR unknown_cmd\n"
  end
end

-- Make the (otherwise blocking) receive time out quickly so the frame loop
-- never stalls. On timeout, socketServerResponse() returns "" which the loop
-- guard below treats as "no command this frame".
comm.socketServerSetTimeout(1)

-- Un-throttle emulation. The loop only advances a frame per "step" command, but
-- BizHawk still throttles frameadvance to the configured speed (100% = 60fps),
-- which makes a full ES generation take hours. Set this high for training; drop
-- it to 100 if you want to WATCH the run in real time.
local EMULATOR_SPEED_PERCENT = 3200
client.speedmode(EMULATOR_SPEED_PERCENT)

-- One-time diagnostic: dump every control name the core exposes for controller
-- 1 so we can confirm the exact analog axis key strings (they vary by core /
-- Guncon binding). If the aim axes below silently do nothing, read this dump in
-- the Lua console and fix GUNCON_AIM_X_KEY / GUNCON_AIM_Y_KEY to match.
do
  local ok, state = pcall(joypad.get, 1)
  if ok and type(state) == "table" then
    print("[bridge] controller 1 controls:")
    for name, value in pairs(state) do
      print(string.format("  %-18s = %s", tostring(name), tostring(value)))
    end
  else
    print("[bridge] joypad.get(1) diagnostic failed: " .. tostring(state))
  end
end

-- Announce readiness exactly once so Python's connect() can stop blocking and
-- know the script is live and able to service commands.
print("======================================================")
print("[bridge] *** SCRIPT RELOADED -- VERSION WITH READ-BACK DEBUG ***")
print("[bridge] DEBUG_INPUT_LOG = " .. tostring(DEBUG_INPUT_LOG))
print("======================================================")
comm.socketServerSend("READY\n")
print("[bridge] sent READY -- waiting for commands")

-- CRITICAL: advance one frame RIGHT NOW to pump the comm transport and flush
-- the READY bytes out to Python. Without this, we would enter the loop and
-- block on socketServerResponse() before READY is ever sent -> deadlock
-- (Python waits for READY, Lua waits for a command) and BizHawk freezes.
emu.frameadvance()

-- Main loop.
-- CRITICAL ORDERING: the frame advance MUST happen before reading the next
-- socket command.  The `step` handler responds "OK" immediately (before the
-- frame is actually advanced), so at high emulation speed Python can send the
-- next tick's set_input before we've called apply_input() for the current tick.
-- If we read the socket first (old order), that next set_input overwrites
-- cover/shoot and the wrong values get applied -- producing 1-frame spam.
-- Solution: check pending_steps at the TOP of the loop; only read the socket
-- when there is nothing left to advance.
while true do
  if pending_steps > 0 then
    -- Advance the queued frame BEFORE reading any new commands.
    draw_hud()
    apply_input()
    pending_steps = pending_steps - 1
    emu.frameadvance()
  else
    -- No frame pending: read and service exactly one socket command, then yield.
    local line = comm.socketServerResponse()
    if line and line ~= "" then
      local ok, resp = pcall(handle, line)
      comm.socketServerSend(ok and resp or ("ERR " .. tostring(resp) .. "\n"))
    end
    draw_hud()
    emu.yield()
  end
end
