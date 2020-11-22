parameters = {

'diff_privacy': False,
'sigma': 0.01,

# aggregation_methods Allowed values are: [ 'mean', 'geom_median','foolsgold']
'aggregation_methods': 'mean',


#global lr:
'eta': 0.1,  # single-shot: 0.1; multi-shot: 1

'baseline': False, # single-shot: false; multi-shot:true


# distributed attackers: (training img num : 606 - 591 - 568 - 557)
#'adversary_list': [41, 73, 51, 74],
# centralized attacker: (training img num :602)
#'adversary_list': [95],

#smaller size distributed attacker:
'adversary_list': [51, 74],

'trigger_num': 2,
#'trigger_num': 4,

## gap 2 size 1*4 base (0, 0)
#'0_poison_pattern': [[0, 0], [0, 1], [0, 2], [0, 3]],
#'1_poison_pattern': [[0, 6], [0, 7], [0, 8], [0, 9]],
#'2_poison_pattern': [[3, 0], [3, 1], [3, 2], [3, 3]],
#'3_poison_pattern': [[3, 6], [3, 7], [3, 8], [3, 9]],


#0_poison_pattern: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 6], [0, 7], [0, 8], [0, 9],
#                   [3, 0], [3, 1], [3, 2], [3, 3], [3, 6], [3, 7], [3, 8], [3, 9]]
#1_poison_pattern: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 6], [0, 7], [0, 8], [0, 9],
#                   [3, 0], [3, 1], [3, 2], [3, 3], [3, 6], [3, 7], [3, 8], [3, 9]]
#2_poison_pattern: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 6], [0, 7], [0, 8], [0, 9],
#                  [3, 0], [3, 1], [3, 2], [3, 3], [3, 6], [3, 7], [3, 8], [3, 9]]
#3_poison_pattern: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 6], [0, 7], [0, 8], [0, 9],
#                   [3, 0], [3, 1], [3, 2], [3, 3], [3, 6], [3, 7], [3, 8], [3, 9]]                                      


# Influence of number of pixels
#0_poison_pattern: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]]
#1_poison_pattern: [[0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [0, 15], [0, 16], [0, 17]]
#2_poison_pattern: [[3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7]]
#_poison_pattern: [[3, 10], [3, 11], [3, 12], [3, 13], [3, 14], [3, 15], [3, 16], [3, 17]]


'0_poison_pattern': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 6], [0, 7], [0, 8], [0, 9]],
'1_poison_pattern': [[3, 0], [3, 1], [3, 2], [3, 3], [3, 6], [3, 7], [3, 8], [3, 9]],

# single shot - distributed attack:
#'0_poison_epochs': [12],
#'1_poison_epochs': [14],
#'2_poison_epochs': [16],
#'3_poison_epochs': [18]

'0_poison_epochs': [16],
'1_poison_epochs': [18],

#single shot - centralized attack:
# 0_poison_epochs: [18]


# multi-shot:
#'0_poison_epochs': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,100],
#'1_poison_epochs': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,100],
#'2_poison_epochs': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,100],
#'3_poison_epochs': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,100]

}
