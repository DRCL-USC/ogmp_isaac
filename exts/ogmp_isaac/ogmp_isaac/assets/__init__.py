from ogmplm.assets.objects.goalposts.goalposts import GOAL_DEPTH, GOALPOSTS_CFG
from ogmplm.assets.objects.platform.platform import PLATFORM_CFG
from ogmplm.assets.robots.berkeley_humanoid.biped.berkeley_humanoid import BERKELEY_HUMANOID_CFG
from ogmplm.assets.robots.G1.g1_cfg import G1_CFG, G1_DC_CFG, G1_MINIMAL_CFG, G1_SPLIT_CFG
from ogmplm.assets.robots.H1.h1_cfg import H1_CFG, H1_DC_CFG, H1_MINIMAL_CFG
from ogmplm.assets.robots.hector.v1.wo_coupling import HECTOR_V1_CFG, HECTOR_V1_DC_CFG
from ogmplm.assets.robots.hector.v1p5.w_coupling import HECTOR_V1P5_CFG, HECTOR_V1P5_DC_CFG, HECTOR_V1P5_IPD_CFG

ROBOTS = {
    "HECTOR_V1": HECTOR_V1_CFG,
    "HECTOR_V1_DC": HECTOR_V1_DC_CFG,
    "HECTOR_V1P5": HECTOR_V1P5_CFG,
    "HECTOR_V1P5_IPD": HECTOR_V1P5_IPD_CFG,
    "HECTOR_V1P5_DC": HECTOR_V1P5_DC_CFG,
    "BERKELEY_HUMANOID": BERKELEY_HUMANOID_CFG,
    "G1": G1_CFG,
    "G1_DC": G1_DC_CFG,
    "G1_MINIMAL": G1_MINIMAL_CFG,
    "G1_SPLIT": G1_SPLIT_CFG,
    "H1": H1_CFG,
    "H1_DC": H1_DC_CFG,
    "H1_MINIMAL": H1_MINIMAL_CFG,
}
