// SPDX-License-Identifier: MIT
pragma solidity 0.8.26;

import {Test, stdJson} from "forge-std/Test.sol";
import {OracleAnchoredLVRHook} from "src/OracleAnchoredLVRHook.sol";
import {ChainlinkReferenceOracle} from "src/oracles/ChainlinkReferenceOracle.sol";
import {Deployers} from "../../lib/v4-core/test/utils/Deployers.sol";
import {IHooks} from "v4-core/interfaces/IHooks.sol";
import {IPoolManager} from "v4-core/interfaces/IPoolManager.sol";
import {ModifyLiquidityParams} from "v4-core/types/PoolOperation.sol";
import {Hooks} from "v4-core/libraries/Hooks.sol";
import {LPFeeLibrary} from "v4-core/libraries/LPFeeLibrary.sol";
import {FullMath} from "v4-core/libraries/FullMath.sol";
import {TickMath} from "v4-core/libraries/TickMath.sol";
import {ManualAggregatorV3} from "../helpers/ManualAggregatorV3.sol";

contract OracleAnchoredLVRHookProductionEvaluator is Test, Deployers {
    using stdJson for string;

    error ToxicScenarioDidNotQuoteToxic(int24 gapTicks, bool zeroForOne, uint24 feeUnits);
    error UnexpectedPreviewFailure(int24 gapTicks, bytes revertData);
    error UnexpectedWidthFailure(uint256 sigma2PerSecondWad, uint32 latencySecs, uint256 lvrBudgetWad, bytes revertData);

    uint256 internal constant WAD = 1e18;
    uint256 internal constant SQRT_WAD = 1e9;
    uint256 internal constant BPS_DENOMINATOR = 10_000;
    uint24 internal constant BASE_FEE = 500;
    uint24 internal constant MAX_FEE = 50_000;
    uint24 internal constant ALPHA_BPS = 10_000;
    uint32 internal constant MAX_ORACLE_AGE = 1 hours;
    uint32 internal constant LATENCY_SECS = 60;
    uint32 internal constant CENTER_TOLERANCE_TICKS = 30;
    uint256 internal constant LVR_BUDGET_WAD = 1e16;
    uint256 internal constant DEFAULT_SIGMA2_PER_SECOND_WAD = 4e14;
    uint256 internal constant BOOTSTRAP_SIGMA2_PER_SECOND_WAD = 8e14;
    uint256 internal constant OFFERED_WIDTH_TICKS = 24_000;
    uint256 internal constant ECONOMIC_FEE_CEILING_BPS = 9_000;
    string internal constant METRICS_DIR = "/.harness/foundry_hook";
    string internal constant METRICS_PATH = "/.harness/foundry_hook/metrics.json";

    OracleAnchoredLVRHook internal hook;
    ChainlinkReferenceOracle internal oracle;
    ManualAggregatorV3 internal baseFeed;
    ManualAggregatorV3 internal quoteFeed;

    function setUp() public {
        deployFreshManagerAndRouters();
        deployMintAndApprove2Currencies();

        address hookAddress = _permissionedHookAddress();
        deployCodeTo(
            "src/OracleAnchoredLVRHook.sol:OracleAnchoredLVRHook",
            abi.encode(IPoolManager(manager), address(this)),
            hookAddress
        );
        hook = OracleAnchoredLVRHook(hookAddress);

        baseFeed = new ManualAggregatorV3(18, int256(WAD), block.timestamp);
        quoteFeed = new ManualAggregatorV3(18, int256(WAD), block.timestamp);
        oracle = new ChainlinkReferenceOracle(baseFeed, false, quoteFeed, false, 18, 18);

        (key,) = initPool(
            currency0,
            currency1,
            IHooks(address(hook)),
            LPFeeLibrary.DYNAMIC_FEE_FLAG,
            SQRT_PRICE_1_1
        );

        hook.setConfig(key, _defaultConfig());
        hook.setRiskState(key, DEFAULT_SIGMA2_PER_SECOND_WAD, WAD, block.timestamp);
        _addLiquidity(-18_000, 18_000, bytes32("seed"));
    }

    function test_writesHarnessMetrics() public {
        FeeMetrics memory feeMetrics = _evaluateFeeGrid();
        WidthMetrics memory widthMetrics = _evaluateWidthGrid();
        uint256 widthPenaltyWad = _widthPenaltyWad(feeMetrics.grossLvrWad, widthMetrics);
        uint256 productionObjectiveWad =
            feeMetrics.productionShortfallWad + feeMetrics.feeOvershootPenaltyWad + widthPenaltyWad;

        string memory metrics = "metrics";
        metrics.serialize("gross_lvr_wad", feeMetrics.grossLvrWad);
        metrics.serialize("quoted_fee_revenue_wad", feeMetrics.quotedFeeRevenueWad);
        metrics.serialize("realized_fee_revenue_wad", feeMetrics.realizedFeeRevenueWad);
        metrics.serialize("quoted_unrecaptured_lvr_wad", feeMetrics.quotedUnrecapturedLvrWad);
        metrics.serialize("unrecaptured_lvr_wad", feeMetrics.productionShortfallWad);
        metrics.serialize("lp_net_from_toxic_flow_wad", feeMetrics.realizedLpNetFromToxicFlowWad);
        metrics.serialize("quoted_lp_net_from_toxic_flow_wad", feeMetrics.quotedLpNetFromToxicFlowWad);
        metrics.serialize("quoted_recapture_ratio_bps", feeMetrics.quotedRecaptureRatioBps);
        metrics.serialize("recapture_ratio_bps", feeMetrics.effectiveRecaptureRatioBps);
        metrics.serialize("effective_recapture_ratio_bps", feeMetrics.effectiveRecaptureRatioBps);
        metrics.serialize("production_shortfall_wad", feeMetrics.productionShortfallWad);
        metrics.serialize("fee_overshoot_penalty_wad", feeMetrics.feeOvershootPenaltyWad);
        metrics.serialize("production_objective_wad", productionObjectiveWad);
        metrics.serialize("avg_toxic_fee_units", feeMetrics.avgToxicFeeUnits);
        metrics.serialize("max_toxic_fee_units", feeMetrics.maxToxicFeeUnits);
        metrics.serialize("toxic_event_count", feeMetrics.toxicEventCount);
        metrics.serialize("executed_toxic_event_count", feeMetrics.executedToxicEventCount);
        metrics.serialize("economically_executable_event_count", feeMetrics.economicallyExecutableEventCount);
        metrics.serialize("economic_execution_rate_bps", feeMetrics.economicExecutionRateBps);
        metrics.serialize("cap_hit_count", feeMetrics.capHitCount);
        metrics.serialize("cap_hit_rate_bps", feeMetrics.capHitRateBps);
        metrics.serialize("gap_scenario_count", feeMetrics.gapScenarioCount);
        metrics.serialize("economic_fee_ceiling_bps", ECONOMIC_FEE_CEILING_BPS);
        metrics.serialize("width_penalty_wad", widthPenaltyWad);
        metrics.serialize("admission_rate_bps", widthMetrics.admissionRateBps);
        metrics.serialize("mean_required_width_ticks", widthMetrics.meanRequiredWidthTicks);
        metrics.serialize("max_required_width_ticks", widthMetrics.maxRequiredWidthTicks);
        metrics.serialize("width_scenario_count", widthMetrics.scenarioCount);
        metrics.serialize("impossible_budget_count", widthMetrics.impossibleBudgetCount);
        metrics.serialize("offered_width_ticks", OFFERED_WIDTH_TICKS);
        metrics.serialize("default_sigma2_per_second_wad", DEFAULT_SIGMA2_PER_SECOND_WAD);
        metrics.serialize("default_latency_secs", LATENCY_SECS);
        string memory metricsJson = metrics.serialize("default_lvr_budget_wad", LVR_BUDGET_WAD);

        string memory payload = "payload";
        string memory finalJson = payload.serialize("metrics", metricsJson);

        string memory metricsDir = string.concat(vm.projectRoot(), METRICS_DIR);
        vm.createDir(metricsDir, true);
        finalJson.write(string.concat(vm.projectRoot(), METRICS_PATH));
    }

    struct FeeMetrics {
        uint256 grossLvrWad;
        uint256 quotedFeeRevenueWad;
        uint256 realizedFeeRevenueWad;
        int256 quotedUnrecapturedLvrWad;
        int256 quotedLpNetFromToxicFlowWad;
        int256 realizedLpNetFromToxicFlowWad;
        uint256 quotedRecaptureRatioBps;
        uint256 effectiveRecaptureRatioBps;
        uint256 productionShortfallWad;
        uint256 feeOvershootPenaltyWad;
        uint256 avgToxicFeeUnits;
        uint256 maxToxicFeeUnits;
        uint256 toxicEventCount;
        uint256 executedToxicEventCount;
        uint256 economicallyExecutableEventCount;
        uint256 economicExecutionRateBps;
        uint256 capHitCount;
        uint256 capHitRateBps;
        uint256 gapScenarioCount;
    }

    struct WidthMetrics {
        uint256 admissionRateBps;
        uint256 meanRequiredWidthTicks;
        uint256 maxRequiredWidthTicks;
        uint256 scenarioCount;
        uint256 impossibleBudgetCount;
    }

    function _evaluateFeeGrid() internal returns (FeeMetrics memory metrics) {
        uint256 totalFeeUnits = 0;

        for (int24 absoluteGap = 10; absoluteGap <= 800; absoluteGap = _nextGap(absoluteGap)) {
            metrics.gapScenarioCount += 1;

            for (uint256 sign = 0; sign < 2; ++sign) {
                int24 signedGap = sign == 0 ? absoluteGap : -absoluteGap;
                bool zeroForOne = signedGap < 0;
                uint256 toxicNotionalWad = _toxicInputNotionalWad(signedGap);
                uint256 grossEventWad = FullMath.mulDiv(toxicNotionalWad, toxicNotionalWad, WAD);

                metrics.toxicEventCount += 1;
                metrics.grossLvrWad += grossEventWad;
                _setOraclePrice(_priceWadAtTick(signedGap), block.timestamp);

                try hook.previewSwapFee(key, zeroForOne) returns (
                    bool toxic,
                    uint24 feeUnits,
                    uint160,
                    uint160
                ) {
                    if (!toxic) {
                        revert ToxicScenarioDidNotQuoteToxic(signedGap, zeroForOne, feeUnits);
                    }

                    uint256 quotedFeeWad = uint256(feeUnits) * 1e12;
                    uint256 quotedFeeRevenueWad =
                        FullMath.mulDiv(quotedFeeWad, toxicNotionalWad, WAD);
                    uint256 economicFeeCeilingWad =
                        FullMath.mulDiv(toxicNotionalWad, ECONOMIC_FEE_CEILING_BPS, BPS_DENOMINATOR);
                    uint256 realizedFeeWad =
                        quotedFeeWad < economicFeeCeilingWad ? quotedFeeWad : economicFeeCeilingWad;
                    uint256 realizedFeeRevenueWad =
                        FullMath.mulDiv(realizedFeeWad, toxicNotionalWad, WAD);

                    metrics.executedToxicEventCount += 1;
                    if (quotedFeeWad <= economicFeeCeilingWad) {
                        metrics.economicallyExecutableEventCount += 1;
                    }
                    totalFeeUnits += feeUnits;
                    if (feeUnits > metrics.maxToxicFeeUnits) {
                        metrics.maxToxicFeeUnits = feeUnits;
                    }
                    metrics.quotedFeeRevenueWad += quotedFeeRevenueWad;
                    metrics.realizedFeeRevenueWad += realizedFeeRevenueWad;
                    if (quotedFeeRevenueWad > realizedFeeRevenueWad) {
                        metrics.feeOvershootPenaltyWad += quotedFeeRevenueWad - realizedFeeRevenueWad;
                    }
                } catch (bytes memory revertData) {
                    if (!_matchesSelector(revertData, OracleAnchoredLVRHook.DeviationTooLarge.selector)) {
                        revert UnexpectedPreviewFailure(signedGap, revertData);
                    }
                    metrics.capHitCount += 1;
                }
            }
        }

        metrics.quotedUnrecapturedLvrWad =
            int256(metrics.grossLvrWad) - int256(metrics.quotedFeeRevenueWad);
        metrics.quotedLpNetFromToxicFlowWad =
            int256(metrics.quotedFeeRevenueWad) - int256(metrics.grossLvrWad);
        metrics.realizedLpNetFromToxicFlowWad =
            int256(metrics.realizedFeeRevenueWad) - int256(metrics.grossLvrWad);
        metrics.quotedRecaptureRatioBps = metrics.grossLvrWad == 0
            ? 0
            : FullMath.mulDiv(metrics.quotedFeeRevenueWad, BPS_DENOMINATOR, metrics.grossLvrWad);
        metrics.effectiveRecaptureRatioBps = metrics.grossLvrWad == 0
            ? 0
            : FullMath.mulDiv(metrics.realizedFeeRevenueWad, BPS_DENOMINATOR, metrics.grossLvrWad);
        metrics.productionShortfallWad = metrics.grossLvrWad > metrics.realizedFeeRevenueWad
            ? metrics.grossLvrWad - metrics.realizedFeeRevenueWad
            : 0;
        metrics.avgToxicFeeUnits =
            metrics.executedToxicEventCount == 0 ? 0 : totalFeeUnits / metrics.executedToxicEventCount;
        metrics.economicExecutionRateBps = metrics.toxicEventCount == 0
            ? 0
            : FullMath.mulDiv(
                metrics.economicallyExecutableEventCount, BPS_DENOMINATOR, metrics.toxicEventCount
            );
        metrics.capHitRateBps = metrics.toxicEventCount == 0
            ? 0
            : FullMath.mulDiv(metrics.capHitCount, BPS_DENOMINATOR, metrics.toxicEventCount);
    }

    function _evaluateWidthGrid() internal returns (WidthMetrics memory metrics) {
        uint256[4] memory sigma2Grid = [
            uint256(160_000_000_000),
            uint256(640_000_000_000),
            uint256(2_560_000_000_000),
            uint256(10_240_000_000_000)
        ];
        uint32[4] memory latencyGrid = [uint32(12), 30, 60, 120];
        uint256[3] memory budgetGrid = [uint256(5e15), 1e16, 2e16];
        uint256 admittedCount = 0;
        uint256 widthSum = 0;
        uint256 successfulScenarios = 0;

        for (uint256 sigmaIndex = 0; sigmaIndex < sigma2Grid.length; ++sigmaIndex) {
            for (uint256 latencyIndex = 0; latencyIndex < latencyGrid.length; ++latencyIndex) {
                for (uint256 budgetIndex = 0; budgetIndex < budgetGrid.length; ++budgetIndex) {
                    metrics.scenarioCount += 1;

                    OracleAnchoredLVRHook.Config memory cfg = _defaultConfig();
                    cfg.latencySecs = latencyGrid[latencyIndex];
                    cfg.lvrBudgetWad = budgetGrid[budgetIndex];
                    hook.setConfig(key, cfg);
                    hook.setRiskState(key, sigma2Grid[sigmaIndex], WAD, block.timestamp);

                    try hook.minWidthTicks(key) returns (uint256 requiredWidthTicks) {
                        successfulScenarios += 1;
                        widthSum += requiredWidthTicks;
                        if (requiredWidthTicks > metrics.maxRequiredWidthTicks) {
                            metrics.maxRequiredWidthTicks = requiredWidthTicks;
                        }
                        if (requiredWidthTicks <= OFFERED_WIDTH_TICKS) {
                            admittedCount += 1;
                        }
                    } catch (bytes memory revertData) {
                        if (!_matchesSelector(revertData, OracleAnchoredLVRHook.ImpossibleBudget.selector)) {
                            revert UnexpectedWidthFailure(
                                sigma2Grid[sigmaIndex],
                                latencyGrid[latencyIndex],
                                budgetGrid[budgetIndex],
                                revertData
                            );
                        }
                        metrics.impossibleBudgetCount += 1;
                    }
                }
            }
        }

        metrics.meanRequiredWidthTicks = successfulScenarios == 0 ? 0 : widthSum / successfulScenarios;
        metrics.admissionRateBps =
            metrics.scenarioCount == 0 ? 0 : FullMath.mulDiv(admittedCount, BPS_DENOMINATOR, metrics.scenarioCount);
    }

    function _widthPenaltyWad(uint256 grossLvrWad, WidthMetrics memory widthMetrics)
        internal
        pure
        returns (uint256)
    {
        uint256 widthPenaltyWad = widthMetrics.admissionRateBps >= BPS_DENOMINATOR
            ? 0
            : FullMath.mulDiv(grossLvrWad, BPS_DENOMINATOR - widthMetrics.admissionRateBps, BPS_DENOMINATOR);

        if (widthMetrics.scenarioCount > 0 && widthMetrics.impossibleBudgetCount > 0) {
            widthPenaltyWad +=
                FullMath.mulDiv(grossLvrWad, widthMetrics.impossibleBudgetCount, widthMetrics.scenarioCount);
        }

        return widthPenaltyWad;
    }

    function _defaultConfig() internal view returns (OracleAnchoredLVRHook.Config memory cfg) {
        cfg = OracleAnchoredLVRHook.Config({
            oracle: oracle,
            baseFee: BASE_FEE,
            maxFee: MAX_FEE,
            alphaBps: ALPHA_BPS,
            maxOracleAge: MAX_ORACLE_AGE,
            latencySecs: LATENCY_SECS,
            centerTolTicks: CENTER_TOLERANCE_TICKS,
            lvrBudgetWad: LVR_BUDGET_WAD,
            bootstrapSigma2PerSecondWad: BOOTSTRAP_SIGMA2_PER_SECOND_WAD
        });
    }

    function _permissionedHookAddress() internal view returns (address) {
        uint160 permissions = Hooks.BEFORE_ADD_LIQUIDITY_FLAG | Hooks.BEFORE_SWAP_FLAG;
        uint160 mask = uint160(type(uint160).max) & clearAllHookPermissionsMask;
        return address(uint160(mask | permissions));
    }

    function _addLiquidity(int24 tickLower, int24 tickUpper, bytes32 salt) internal {
        modifyLiquidityRouter.modifyLiquidity(
            key,
            ModifyLiquidityParams({tickLower: tickLower, tickUpper: tickUpper, liquidityDelta: 1e18, salt: salt}),
            ZERO_BYTES
        );
    }

    function _setOraclePrice(uint256 priceWad, uint256 updatedAt) internal {
        baseFeed.setRoundData(int256(priceWad), updatedAt);
        quoteFeed.setRoundData(int256(WAD), updatedAt);
    }

    function _toxicInputNotionalWad(int24 gapTicks) internal pure returns (uint256) {
        int24 absoluteTicks = gapTicks >= 0 ? gapTicks : -gapTicks;
        uint160 referenceSqrtPriceX96 = TickMath.getSqrtPriceAtTick(absoluteTicks);
        return FullMath.mulDiv(referenceSqrtPriceX96, WAD, SQRT_PRICE_1_1) - WAD;
    }

    function _priceWadAtTick(int24 tick) internal pure returns (uint256) {
        uint160 sqrtPriceX96 = TickMath.getSqrtPriceAtTick(tick);
        uint256 sqrtPriceWad = FullMath.mulDiv(sqrtPriceX96, SQRT_WAD, 2 ** 96);
        return FullMath.mulDiv(sqrtPriceWad, sqrtPriceWad, 1);
    }

    function _nextGap(int24 absoluteGap) internal pure returns (int24) {
        if (absoluteGap < 100) {
            return absoluteGap + 10;
        }
        if (absoluteGap < 400) {
            return absoluteGap + 25;
        }
        return absoluteGap + 50;
    }

    function _matchesSelector(bytes memory revertData, bytes4 selector) internal pure returns (bool) {
        if (revertData.length < 4) {
            return false;
        }

        bytes4 actualSelector;
        assembly {
            actualSelector := mload(add(revertData, 32))
        }
        return actualSelector == selector;
    }
}
