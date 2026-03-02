const AGENTS = [:normal, :perfect, :random, :naive, :scaled, :noisy] 

const TG_PROBS = [:tag,
    :tag_inf_0_1_2_5_10, :tag_inf_0_1_2_5_10_w05,
    :tag_inf_1, :tag_inf_2, :tag_inf_5,
    :tag_1_5,
    :tag_1_0_1_2_5_10, :tag_1_0_1_2_5_10_w05
]
    
const TG_ASK_PROBS = [
    :tag_inf_0_1_2_5_10, :tag_inf_0_1_2_5_10_w05,
    :tag_inf_1, :tag_inf_2, :tag_inf_5, :tag_inf_10,
    :tag_1_5,
    :tag_1_0_1_2_5_10, :tag_1_0_1_2_5_10_w05
]

const RS_PROBS = [
    :rs84, :rs84_inf_1, :rs84_inf_2, :rs84_inf_5, :rs84_1_5, 
    :rs84_inf_0_1_2_5_10, :rs84_inf_0_1_2_5_10_w05,
    :rs78, :rs78_inf_1, :rs78_inf_2, :rs78_inf_5, 
    :rs78_inf_0_1_2_5_10, :rs78_inf_0_1_2_5_10_w05
] 

const RS_ASK_PROBS = [
    :rs84_inf_1, :rs84_inf_2, :rs84_inf_5, 
    :rs84_inf_0_1_2_5_10, :rs84_inf_0_1_2_5_10_w05,
    :rs78_inf_1, :rs78_inf_2, :rs78_inf_5,
    :rs78_inf_0_1_2_5_10, :rs78_inf_0_1_2_5_10_w05
]
