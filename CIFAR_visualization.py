
import argparse
import os
import datetime
import matplotlib.pyplot as plt 

mode_list = ["cen", "multi_cen", "dis", "double_pix", "defense", "half_attack", "similarity", "all"]

target_file_name = "posiontest_result.csv"
main_file_name = "test_result.csv"


cifar_figure_folder = "./saved_figures/cifar/"

color = ['lightseagreen',
'indianred',
'dimgray',
'tab:blue',
'blueviolet',
'black'
]


MarkerSize = 3
def get_newest_folder(base_folder, compare_folder):
    base_dirs = os.listdir(base_folder)
    compare_dirs = os.listdir(compare_folder)

    max_base_date = base_dirs[0]
    max_compare_date = compare_dirs[0]


    for dirs in base_dirs:
        if dirs > max_base_date:
            max_base_date = dirs

    for dirs in compare_dirs:
        if dirs > max_compare_date:
            max_compare_date = dirs
    
    return os.path.join(base_folder, max_base_date), os.path.join(compare_folder, max_compare_date)


def draw_figure(folder1, folder2, name_list):
    plt.figure()
    color_index = 0
    
    for file_idx, file_name in enumerate([target_file_name, main_file_name]):
        for folder_idx, folder_name in enumerate([folder1, folder2]):
            filepath = folder_name + "/" + file_name
            record_dict = {}
            with open(filepath, 'r') as f:
                rl = f.readlines()
                for line in rl:
                    if "global" in line:
                        record_dict[int(line.split(",")[1])] = float(line.split(",")[-3])
            label_name = "_target" if file_idx == 0 else "_main"
            values = [record_dict[iteration] for iteration in record_dict.keys()]
            plt.plot(record_dict.keys(), values, label = name_list[folder_idx] + label_name, marker = 'o', color = color[color_index], markersize = MarkerSize)
            color_index += 1
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.savefig(cifar_figure_folder + name_list[0] + "_" + name_list[1] + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mode')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()
    
    mode = args.params
    check_correctness = False

    dis_sing_folder = "./saved_models/cifar_dis_sing/"
    dis_multi_folder = "./saved_models/cifar_dis_multi/"

    if mode == "cen" or mode == "all":
        cen_sing_folder = "./saved_models/cifar_cen_sing/"
        cen_multi_folder = "./saved_models/cifar_cen_multi/"

        base_folder, target_folder = get_newest_folder(dis_sing_folder, cen_sing_folder)
        draw_figure(base_folder, target_folder, ["dis_single", "cen_single"])

        base_folder, target_folder = get_newest_folder(dis_multi_folder, cen_multi_folder)
        draw_figure(base_folder, target_folder, ["dis_multi", "cen_multi"])

    if mode == "multi_cen" or mode == "all":
        cen_sing_folder = "./saved_models/cifar_multi_cen_sing/"
        cen_multi_folder = "./saved_models/cifar_multi_cen_multi/"

        base_folder, target_folder = get_newest_folder(dis_sing_folder, cen_sing_folder)
        draw_figure(base_folder, target_folder, ["dis_single", "multi_cen_single"])

        base_folder, target_folder = get_newest_folder(dis_multi_folder, cen_multi_folder)
        draw_figure(base_folder, target_folder, ["dis_multi", "multi_cen_multi"])


    if mode == "double_pix" or mode == "all":
        cen_sing_folder = "./saved_models/cifar_double_pix_sing/"
        cen_multi_folder = "./saved_models/cifar_double_pix_multi/"

        base_folder, target_folder = get_newest_folder(dis_sing_folder, cen_sing_folder)
        draw_figure(base_folder, target_folder, ["dis_single", "double_pix_single"])

        base_folder, target_folder = get_newest_folder(dis_multi_folder, cen_multi_folder)
        draw_figure(base_folder, target_folder, ["dis_multi", "double_pix_multi"])


    if mode == "half_attack" or mode == "all":
        cen_sing_folder = "./saved_models/cifar_half_attack_sing/"
        cen_multi_folder = "./saved_models/cifar_half_attack_multi/"

        base_folder, target_folder = get_newest_folder(dis_sing_folder, cen_sing_folder)
        draw_figure(base_folder, target_folder, ["dis_single", "half_attack_single"])

        base_folder, target_folder = get_newest_folder(dis_multi_folder, cen_multi_folder)
        draw_figure(base_folder, target_folder, ["dis_multi", "half_attack_multi"])

    if mode == "defense" or mode == "all":
        cen_sing_folder = "./saved_models/cifar_foolsgold/"
        cen_multi_folder = "./saved_models/cifar_geomedian/"

        base_folder, target_folder = get_newest_folder(dis_sing_folder, cen_sing_folder)
        draw_figure(base_folder, target_folder, ["dis_single", "foolsgold"])

        base_folder, target_folder = get_newest_folder(dis_multi_folder, cen_multi_folder)
        draw_figure(base_folder, target_folder, ["dis_multi", "geomedian"])
    
    


    





