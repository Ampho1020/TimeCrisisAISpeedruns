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
-- Commands (one per line):
--   read_u16 <addr>                               -> OK <value>
--   set_input <shoot01> <cover01> <aim_x> <aim_y> -> OK   (aim_* optional)
--   step <n>                                      -> OK
--   load <slot> / save <slot>                     -> OK
--   frame                                         -> OK <framecount>
--   hud <line1|line2|...> / hud_clear             -> OK
--
-- NOTES
--   * Aim X/Y are accepted and stored but intentionally NOT written to the
--     Guncon axes yet -- the in-game lightgun calibration is baked into the
--     savestate, and real aiming lands with the vision step. Only the trigger
--     (shoot) and P1 A (cover) buttons are driven for now. This keeps the
--     per-frame path minimal so nothing out-of-range can fault the core.
--   * Confirmed Guncon key names on this build:
--       trigger = "P1 Trigger" (left mouse / shoot)
--       cover   = "P1 A"       (right mouse)
--       axes    = "P1 X Axis" (0..2640), "P1 Y Axis" (16..256)  [unused for now]

local DOMAIN = "MainRAM"

local shoot, cover = false, false
local aim_x_norm, aim_y_norm = 0.5, 0.5   -- stored only; not written yet
local hud_lines = {}

-- pending frames to advance, consumed one-per-iteration by the main loop
local pending_steps = 0

-- Confirmed Guncon buttons (axes intentionally not written yet).
local GUNCON_TRIGGER_KEY = "P1 Trigger"
local GUNCON_COVER_KEY   = "P1 A"

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
  -- Buttons only for now. Trigger suppressed while covering.
  joypad.set({
    [GUNCON_TRIGGER_KEY] = shoot and not cover,
    [GUNCON_COVER_KEY]   = cover,
  })
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

-- Announce readiness exactly once so Python's connect() can stop blocking and
-- know the script is live and able to service commands.
comm.socketServerSend("READY\n")
print("[bridge] sent READY -- waiting for commands")

-- CRITICAL: advance one frame RIGHT NOW to pump the comm transport and flush
-- the READY bytes out to Python. Without this, we would enter the loop and
-- block on socketServerResponse() before READY is ever sent -> deadlock
-- (Python waits for READY, Lua waits for a command) and BizHawk freezes.
emu.frameadvance()

-- Single, canonical frame loop. Exactly one emu.frameadvance() per iteration.
-- Drain any waiting command, then advance one frame (applying input if a
-- step is pending). We advance every iteration regardless of whether a command
-- arrived, so the comm transport keeps pumping and never deadlocks.
while true do
  local line = comm.socketServerResponse()
  if line and line ~= "" then
    local ok, resp = pcall(handle, line)
    comm.socketServerSend(ok and resp or ("ERR " .. tostring(resp) .. "\n"))
  end

  if pending_steps > 0 then
    apply_input()
    pending_steps = pending_steps - 1
  end

  draw_hud()
  emu.frameadvance()
end
