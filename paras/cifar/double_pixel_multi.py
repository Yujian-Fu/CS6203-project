parameters = {
'diff_privacy': False, 
'sigma': 0.01,

# aggregation_methods Allowed values are: [ 'mean', 'geom_median','foolsgold']
'aggregation_methods': 'mean',

#global lr:
'eta': 1,  # single-shot: 0.1; multi-shot: 1

'baseline': True, # single-shot: false; multi-shot:true

# distributed attackers: (training img num : 526 - 527 - 496 - 546)
'adversary_list': [17, 33, 77, 11],
# centralized attacker: (training img num: 529)
#'adversary_list': [45],

#adversary_list: [77, 11]

'trigger_num': 4,
#trigger_num: 2


# gap 3 size 1*6 base (0, 0)
#0_poison_pattern: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
#1_poison_pattern: [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]]
#2_poison_pattern: [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]]
#3_poison_pattern: [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]

#0_poison_pattern: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], 
#                    [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]
#1_poison_pattern: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], 
#                   [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]
#2_poison_pattern: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], 
#                   [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]
#3_poison_pattern: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], 
#                    [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]

# Influence of the number of pixels
'0_poison_pattern': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11]],
'1_poison_pattern': [[0, 15], [0, 16], [0, 17], [0, 18], [0, 19], [0, 20], [0, 21], [0, 22], [0, 23], [0, 24], [0, 25], [0, 26]],
'2_poison_pattern': [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11]],
'3_poison_pattern': [[4, 15], [4, 16], [4, 17], [4, 18], [4, 19], [4, 20], [4, 21], [4, 22], [4, 23], [4, 24], [4, 25], [4, 26]],

#0_poison_pattern: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]]
#1_poison_pattern: [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]

# single shot - distributed attack:
#0_poison_epochs: [203]
#1_poison_epochs: [205]
#2_poison_epochs: [207]
#3_poison_epochs: [209]
# single shot - centralized attack:
# 0_poison_epochs: [209]

#0_poison_epochs: [207]
#1_poison_epochs: [209]

# multi shot:
'0_poison_epochs': [ 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400],
'1_poison_epochs': [ 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400],
'2_poison_epochs': [ 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400],
'3_poison_epochs': [ 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400],

}