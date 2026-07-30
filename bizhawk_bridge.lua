-- Line-protocol bridge for BizHawk <-> Python ES training.
-- Open via: Tools -> Lua Console -> Open Script
--
-- Commands (one per line):
--   read_u16 <addr>                      -> OK <value>
--   set_input <shoot01> <cover01> <bias> -> OK
--   step <n>                             -> OK
--   load <slot> / save <slot>            -> OK
--   frame                                -> OK <framecount>
--   hud <line1|line2|...> / hud_clear    -> OK

local socket = require("socket")

local HOST, PORT = "127.0.0.1", 8765
local DOMAIN = "MainRAM"

local server = assert(socket.bind(HOST, PORT))
server:settimeout(0)
print(string.format("[bridge] listening on %s:%d", HOST, PORT))

local client = nil
local shoot, cover, aim_bias = false, false, 0.0
local hud_lines = {}

local function parse_int(s)
  if not s then return nil end
  if string.sub(s, 1, 2) == "0x" or string.sub(s, 1, 2) == "0X" then
    return tonumber(s)
  end
  return tonumber(s) or tonumber("0x" .. s)
end

local function draw_hud()
  for i, txt in ipairs(hud_lines) do
    gui.text(4, 4 + (i - 1) * 14, txt, 0xFFFFFF00, 0xC0000000)
  end
end

local function apply_input()
  local inp = {}
  -- IMPORTANT: change these to match your actual BizHawk PS1 bindings.
  -- Check Config -> Controllers, or the Input Display window.
  inp["P1 Cross"] = shoot
  inp["P1 R1"]    = cover
  joypad.set(inp)
  -- aim_bias is accepted but unused until the vision/aiming step is added.
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
    shoot    = (parts[2] == "1")
    cover    = (parts[3] == "1")
    aim_bias = tonumber(parts[4]) or 0.0
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
  if not client then
    client = server:accept()
    if client then
      client:settimeout(0)
      print("[bridge] client connected")
    end
  else
    local line, err = client:receive("*l")
    if line then
      local ok, resp = pcall(handle, line)
      client:send(ok and resp or ("ERR " .. tostring(resp) .. "\n"))
    elseif err == "closed" then
      print("[bridge] client disconnected")
      client:close()
      client = nil
    end
  end
  draw_hud()
  emu.frameadvance()
end
