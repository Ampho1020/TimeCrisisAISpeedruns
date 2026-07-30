-- Line-protocol bridge for BizHawk 2.11.1 comm.* <-> Python ES training.
-- Open via: Tools -> Lua Console -> Open Script
--
-- BizHawk/NLua 2.11.1 does not ship LuaSocket. This script uses BizHawk's
-- native comm.socketServer* API instead. In this transport model BizHawk
-- connects out to a Python listener started by bridge_client.py / es_train.py.
--
-- Commands (one per line):
--   read_u16 <addr>                             -> OK <value>
--   set_input <shoot01> <cover01> <aim_x> <aim_y> -> OK
--   step <n>                                    -> OK
--   load <slot> / save <slot>                   -> OK
--   frame                                       -> OK <framecount>
--   hud <line1|line2|...> / hud_clear           -> OK
--
-- Backward compatibility note:
--   Older clients sent set_input <shoot01> <cover01> <aim_bias>.
--   If aim_y is missing, this script still maps the legacy bias onto a
--   centered X-only placeholder aim so the old command shape does not crash.

local HOST, PORT = "127.0.0.1", 8765
local DOMAIN = "MainRAM"

assert(comm and comm.socketServerSend and comm.socketServerResponse
  and comm.socketServerSetIp and comm.socketServerSetPort,
  "BizHawk 2.11.1 comm.socketServer* API is required")

comm.socketServerSetIp(HOST)
comm.socketServerSetPort(PORT)
print(string.format(
  "[bridge] configured comm target %s:%d (BizHawk connects out to Python)",
  HOST, PORT
))

local GUNCON_TRIGGER_KEY = "TODO_REPLACE_WITH_GUNCON_TRIGGER_KEY"
local GUNCON_AIM_X_KEY   = "TODO_REPLACE_WITH_GUNCON_AIM_X_KEY"
local GUNCON_AIM_Y_KEY   = "TODO_REPLACE_WITH_GUNCON_AIM_Y_KEY"
local GUNCON_COVER_BUTTON_KEY = ""
-- TODO: confirm the exact Guncon axis range on your build. These defaults are
-- a normalized-to-axis mapping placeholder and may need adjustment.
local GUNCON_AXIS_MIN = 0
local GUNCON_AXIS_MAX = 255
-- If your setup ducks by moving the cursor off-screen instead of pressing a
-- dedicated Guncon cover button, these off-screen values are the fallback.
local GUNCON_OFFSCREEN_AXIS_X = GUNCON_AXIS_MIN - 32
local GUNCON_OFFSCREEN_AXIS_Y = GUNCON_AXIS_MIN - 32

local warned_guncon_todo = false
local shoot, cover = false, false
local aim_x_norm, aim_y_norm = 0.5, 0.5
local hud_lines = {}

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

local function normalized_to_axis(v)
  local clamped = clamp01(v)
  return math.floor(GUNCON_AXIS_MIN + clamped * (GUNCON_AXIS_MAX - GUNCON_AXIS_MIN) + 0.5)
end

local function apply_input()
  local inp = {}
  -- DIAGNOSTIC: uncomment this one-shot block to print the exact Guncon key
  -- names that your BizHawk/Nymashock build exposes, then replace the
  -- TODO_REPLACE_* constants at the top of this file with those exact names.
  --[[
  -- if not warned_guncon_todo then
  --   local jp = joypad.get(1) or {}
  --   for k, _ in pairs(jp) do print("[bridge] joypad key: " .. tostring(k)) end
  --   local raw = input.get() or {}
  --   for k, _ in pairs(raw) do print("[bridge] input key: " .. tostring(k)) end
  --   warned_guncon_todo = true
  -- end
  --]]
  if not warned_guncon_todo and (
    string.find(GUNCON_TRIGGER_KEY, "TODO_REPLACE", 1, true)
    or string.find(GUNCON_AIM_X_KEY, "TODO_REPLACE", 1, true)
    or string.find(GUNCON_AIM_Y_KEY, "TODO_REPLACE", 1, true)
  ) then
    print("[bridge] TODO: replace GUNCON_*_KEY constants using the diagnostic block in apply_input()")
    warned_guncon_todo = true
  end

  local axis_x = normalized_to_axis(aim_x_norm)
  local axis_y = normalized_to_axis(aim_y_norm)
  if cover then
    axis_x = GUNCON_OFFSCREEN_AXIS_X
    axis_y = GUNCON_OFFSCREEN_AXIS_Y
  end

  inp[GUNCON_TRIGGER_KEY] = shoot and not cover
  inp[GUNCON_AIM_X_KEY] = axis_x
  inp[GUNCON_AIM_Y_KEY] = axis_y
  if GUNCON_COVER_BUTTON_KEY ~= nil and GUNCON_COVER_BUTTON_KEY ~= "" then
    inp[GUNCON_COVER_BUTTON_KEY] = cover
  end
  joypad.set(inp)
  draw_hud()
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
    if parts[5] ~= nil then
      aim_x_norm = clamp01(tonumber(parts[4]) or 0.5)
      aim_y_norm = clamp01(tonumber(parts[5]) or 0.5)
    else
      local legacy_bias = tonumber(parts[4]) or 0.0
      if legacy_bias < -1.0 then legacy_bias = -1.0 end
      if legacy_bias > 1.0 then legacy_bias = 1.0 end
      aim_x_norm = 0.5 + 0.5 * legacy_bias
      aim_y_norm = 0.5
    end
    return "OK\n"

  elseif cmd == "step" then
    local n = tonumber(parts[2]) or 1
    for _ = 1, n do
      apply_input()
      emu.frameadvance()
    end
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

while true do
  local line = comm.socketServerResponse()
  local advanced_in_handler = false
  if line and line ~= "" then
    local ok, resp = pcall(handle, line)
    if string.sub(line, 1, 5) == "step " or line == "step" then
      advanced_in_handler = true
    end
    comm.socketServerSend(ok and resp or ("ERR " .. tostring(resp) .. "\n"))
  end
  if not advanced_in_handler then
    draw_hud()
    emu.frameadvance()
  end
end
