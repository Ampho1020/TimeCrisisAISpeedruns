-- test_cover_input.lua
-- Standalone cover-input diagnostic for Time Crisis 1 (PS1 / BizHawk 2.11.1).
--
-- PURPOSE:
--   We need to know WHICH input actually controls cover in the emulated game:
--     (A) The "P1 A" digital button (right side of GunCon), OR
--     (B) Pointing the GunCon AIM off-screen (X/Y axis outside valid bounds)
--   The bridge has been using (A), but Time Crisis arcade/PS1 traditionally
--   uses off-screen aim for cover. If (B) is the real mechanism, all our
--   "cover" inputs have been silently ignored.
--
-- HOW TO RUN:
--   1. Load the game and your training savestate FIRST (slot 1).
--   2. Open Tools -> Lua Console -> Open Script -> select this file.
--   3. Watch the Lua console for the printed results.
--   4. Watch the emulator window for the character animation.
--
-- The script pauses after each test phase so you can see what happened.
-- Press "Next phase" (see PAUSE_FRAMES below) or just wait.
--
-- WHAT TO LOOK FOR:
--   "Took damage" or "life changed" in a phase = character was OUT OF COVER.
--   "No damage" = character was protected (IN COVER).
--   "Shots possible" = shots_fired incremented = character can fire.

local DOMAIN       = "MainRAM"
local PHASE_FRAMES = 180   -- ~3 seconds per test phase at 60fps (lower speed to watch)

-- RAM addresses (same as bridge)
local ADDR_LIFE         = 0x0B20C0
local ADDR_SHOTS_FIRED  = 0x0B1F94
local ADDR_TIMER        = 0x0B1D64

-- GunCon keys and ranges (same as bridge)
local TRIGGER_KEY = "P1 Trigger"
local COVER_KEY   = "P1 A"
local AIM_X_KEY   = "P1 X Axis"
local AIM_Y_KEY   = "P1 Y Axis"
local X_MIN, X_MAX = 0, 2640
local Y_MIN, Y_MAX = 16, 256

-- "Off-screen" values to test: move aim well outside valid range.
-- Time Crisis uses off-screen aim to duck in the arcade; test both axes.
local X_CENTER   = math.floor(X_MIN + (X_MAX - X_MIN) * 0.5)   -- 1320
local Y_CENTER   = math.floor(Y_MIN + (Y_MAX - Y_MIN) * 0.5)   -- 136
local X_OFFSCREEN = X_MAX + 500   -- clearly off right edge
local Y_OFFSCREEN = 0             -- below valid range (Y_MIN=16, so 0 is off top)

-- Slow the emulator for visual observation.
client.speedmode(100)

local function read_state()
return {
    life  = memory.read_u16_le(ADDR_LIFE,        DOMAIN),
    shots = memory.read_u16_le(ADDR_SHOTS_FIRED, DOMAIN),
    timer = memory.read_u16_le(ADDR_TIMER,        DOMAIN),
}
end

local function run_phase(label, frames, btn_cover, btn_shoot, aim_x, aim_y)
print(string.format("\n[TEST] Phase: %s  (%d frames)", label, frames))
print(string.format("       P1 A=%s  trigger=%s  aim=(%d,%d)",
                    tostring(btn_cover), tostring(btn_shoot), aim_x, aim_y))

local before = read_state()
print(string.format("       BEFORE  life=%d  shots=%d  timer=%d",
                    before.life, before.shots, before.timer))

for _ = 1, frames do
    joypad.set({
        [TRIGGER_KEY] = btn_shoot,
        [COVER_KEY]   = btn_cover,
    })
    joypad.setanalog({
        [AIM_X_KEY] = aim_x,
        [AIM_Y_KEY] = aim_y,
    })
    emu.frameadvance()
    end

    local after = read_state()
    print(string.format("       AFTER   life=%d  shots=%d  timer=%d",
                        after.life, after.shots, after.timer))

    local life_delta  = after.life  - before.life
    local shots_delta = after.shots - before.shots

    if life_delta < 0 then
        print("  >> DAMAGE TAKEN: " .. math.abs(life_delta) ..
        " -> character was OUT OF COVER (enemies could hit)")
        elseif life_delta == 0 then
            print("  >> NO DAMAGE -> character was IN COVER (or no enemies active yet)")
            end

            if shots_delta > 0 then
                print("  >> SHOTS FIRED: " .. shots_delta ..
                " -> character was OUT OF COVER and trigger accepted")
                else
                    print("  >> NO SHOTS REGISTERED (in cover, or trigger not pressed)")
                    end

                    return after
                    end

                    -- -------------------------------------------------------------------------
                    -- Diagnostic dump: print all controller inputs so we know exact key names.
                    -- -------------------------------------------------------------------------
                    print("\n======================================================")
                    print("[DIAG] Controller 1 inputs (confirm key names):")
                    local ok, state = pcall(joypad.get, 1)
                    if ok and type(state) == "table" then
                        for name, value in pairs(state) do
                            print(string.format("  %-22s = %s", tostring(name), tostring(value)))
                            end
                            else
                                print("  joypad.get(1) failed: " .. tostring(state))
                                end
                                print("======================================================")

                                -- Reload the training savestate so we start from a known game state.
                                savestate.loadslot(1)
                                emu.frameadvance()  -- let the state settle

                                print("\n======================================================")
                                print("TIME CRISIS COVER INPUT DIAGNOSTIC")
                                print("Each phase runs " .. PHASE_FRAMES .. " frames (~3s at 60fps).")
                                print("Watch the character AND the console output.")
                                print("======================================================")

                                -- PHASE 1: Baseline -- do nothing. Check if character takes damage naturally.
                                run_phase("BASELINE (no inputs)",
                                          PHASE_FRAMES, false, false, X_CENTER, Y_CENTER)

                                -- PHASE 2: P1 A held TRUE, aim ON-SCREEN center.
                                -- If P1 A is the cover button, character should be IN COVER (no damage).
                                run_phase("P1_A=true, aim ON-SCREEN",
                                          PHASE_FRAMES, true, false, X_CENTER, Y_CENTER)

                                -- PHASE 3: P1 A held FALSE, aim ON-SCREEN center.
                                -- Without cover, character should take damage (enemies active).
                                run_phase("P1_A=false, aim ON-SCREEN",
                                          PHASE_FRAMES, false, false, X_CENTER, Y_CENTER)

                                -- PHASE 4: P1 A held FALSE, aim OFF-SCREEN (X way off right).
                                -- If off-screen aim triggers cover, character should be protected here.
                                run_phase("P1_A=false, aim OFF-SCREEN (X=" .. X_OFFSCREEN .. ")",
                                          PHASE_FRAMES, false, false, X_OFFSCREEN, Y_CENTER)

                                -- PHASE 5: P1 A held FALSE, aim OFF-SCREEN (Y=0, above valid range).
                                run_phase("P1_A=false, aim OFF-SCREEN (Y=" .. Y_OFFSCREEN .. ")",
                                          PHASE_FRAMES, false, false, X_CENTER, Y_OFFSCREEN)

                                -- PHASE 6: Trigger held with aim ON-SCREEN, P1 A false.
                                -- Should fire shots if character is out of cover.
                                run_phase("TRIGGER=true, P1_A=false, aim ON-SCREEN",
                                          PHASE_FRAMES, false, true, X_CENTER, Y_CENTER)

                                -- PHASE 7: Trigger held with aim ON-SCREEN, P1 A TRUE.
                                -- Tests whether P1 A blocks shooting (if it IS the cover button).
                                run_phase("TRIGGER=true, P1_A=true,  aim ON-SCREEN",
                                          PHASE_FRAMES, true, true, X_CENTER, Y_CENTER)

                                -- PHASE 8: Trigger held with aim OFF-SCREEN (X).
                                -- If off-screen = cover, shots should NOT register despite trigger.
                                run_phase("TRIGGER=true, P1_A=false, aim OFF-SCREEN (X)",
                                          PHASE_FRAMES, false, true, X_OFFSCREEN, Y_CENTER)

                                print("\n======================================================")
                                print("DIAGNOSTIC COMPLETE")
                                print("")
                                print("INTERPRETATION GUIDE:")
                                print("  Phase 2 protects, Phase 3 damages  => P1 A IS the cover button (as assumed)")
                                print("  Phase 4 or 5 protects               => Off-screen aim IS the cover mechanism")
                                print("  Phase 6 fires shots                 => Trigger+onscreen works correctly")
                                print("  Phase 7 blocks shots                => P1 A blocks shooting = cover button")
                                print("  Phase 8 blocks shots                => Off-screen blocks shooting = cover mech")
                                print("")
                                print("If P1 A is NOT cover, update GUNCON_COVER_KEY in bizhawk_bridge.lua,")
                                print("or switch the bridge to use off-screen aim for the cover signal instead.")
                                print("======================================================")
