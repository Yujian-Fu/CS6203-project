
import argparse
import os
import datetime
import matplotlib.pyplot as plt 

mode_list = ["cen", "multi_cen", "dis", "double_pix", "defense", "half_attack", "similarity", "all"]

target_file_name = "posiontest_result.csv"
main_file_name = "test_result.csv"


mnist_figure_folder = "./saved_figures/mnist/"
similarity_figure_folder = './saved_figures/similarity/'

color = ['lightseagreen',
'indianred',
'dimgray',
'tab:blue',
'blueviolet',
'black'
]


def draw_similarity_figure(target_folder, label):
    with open(target_folder + "/model_mean_similarity.txt" , 'r') as f:
        attacker_dis_dict = {}
        attacker_cos_dict = {}

        begnign_dis_dict = {}
        begnign_cos_dict = {}

        weight_list = []

        mode = 0
        prev_name = ""
        rl = f.readlines()
        for line in rl:
            if "Epoch" in line:
                weight_list = list(filter(None,line.split(' ')))[2:-1]
                for weight_name in weight_list:
                    attacker_dis_dict[weight_name] = []
                    begnign_dis_dict[weight_name] = []
                    attacker_cos_dict[weight_name] = []
                    begnign_cos_dict[weight_name] = []
            
            elif "Attacker" in line:
                mode = -1
                
            elif "Begnign" in line:
                mode = 1
            
            else:
                record = list(filter(None,line.split(' ')))[0:-1]
                epoch = int(record[0])

                if record[1] == prev_name:
                    if mode == -1:
                        for idx, parameter_name in enumerate(weight_list):
                            attacker_cos_dict[parameter_name].append([epoch, float(record[idx+2])])
                    else:
                        for idx, parameter_name in enumerate(weight_list):
                            begnign_cos_dict[parameter_name].append([epoch, float(record[idx+2])])

                else:
                    if mode == -1:
                        for idx, parameter_name in enumerate(weight_list):
                            attacker_dis_dict[parameter_name].append([epoch, float(record[idx+2])])
                    else:
                        for idx, parameter_name in enumerate(weight_list):
                            begnign_dis_dict[parameter_name].append([epoch, float(record[idx+2])])

                prev_name = record[1]

        list_x = []
        list_y = []    
        for parameter_name in attacker_cos_dict:
            plt.figure()
            plt.xlabel("Iteration")
            plt.ylabel("Cosine Similarity")
            for item in attacker_cos_dict[parameter_name]:
                list_x.append(item[0])
                list_y.append(item[1] * 0.5 + 0.5)
            plt.scatter(list_x, list_y, color = 'lightseagreen', marker=".", label = "Attacker")
            list_x = []
            list_y = []

            for item in begnign_cos_dict[parameter_name]:
                list_x.append(item[0])
                list_y.append(item[1] * 0.5 + 0.5)
            plt.scatter(list_x, list_y, color = 'indianred', marker=".", label = "Worker")
            plt.legend()
            plt.savefig(similarity_figure_folder + "_" + label + "_" +parameter_name + " Cosine Distance.png")
            plt.close()

        for parameter_name in attacker_dis_dict:
            plt.figure()
            plt.xlabel("Iteration")
            plt.ylabel("Distance Similarity")
            for item in attacker_dis_dict[parameter_name]:
                list_x.append(item[0])
                list_y.append(item[1])
            plt.scatter(list_x, list_y, color = 'lightseagreen', marker=".", label = "Attacker")
            list_x = []
            list_y = []

            for item in begnign_dis_dict[parameter_name]:
                list_x.append(item[0])
                list_y.append(item[1])
            plt.scatter(list_x, list_y, color = 'indianred', marker=".", label = "Worker")
            plt.savefig(similarity_figure_folder + "_" + label + "_" +parameter_name + " Relative Distance.png")
            plt.close()


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
    plt.savefig(mnist_figure_folder + name_list[0] + "_" + name_list[1] + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mode')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()
    
    mode = args.params
    check_correctness = False
    if not os.path.exists(mnist_figure_folder):
        os.makedirs(mnist_figure_folder)

    dis_sing_folder = "./saved_models/mnist_dis_sing/"
    dis_multi_folder = "./saved_models/mnist_dis_multi/"

    if mode == "cen" or mode == "all":
        cen_sing_folder = "./saved_models/mnist_cen_sing/"
        cen_multi_folder = "./saved_models/mnist_cen_multi/"

        base_folder, target_folder = get_newest_folder(dis_sing_folder, cen_sing_folder)
        draw_figure(base_folder, target_folder, ["dis_single", "cen_single"])

        base_folder, target_folder = get_newest_folder(dis_multi_folder, cen_multi_folder)
        draw_figure(base_folder, target_folder, ["dis_multi", "cen_multi"])

    if mode == "multi_cen" or mode == "all":
        cen_sing_folder = "./saved_models/mnist_multi_cen_sing/"
        cen_multi_folder = "./saved_models/mnist_multi_cen_multi/"

        base_folder, target_folder = get_newest_folder(dis_sing_folder, cen_sing_folder)
        draw_figure(base_folder, target_folder, ["dis_single", "multi_cen_single"])

        base_folder, target_folder = get_newest_folder(dis_multi_folder, cen_multi_folder)
        draw_figure(base_folder, target_folder, ["dis_multi", "multi_cen_multi"])


    if mode == "double_pix" or mode == "all":
        cen_sing_folder = "./saved_models/mnist_double_pix_sing/"
        cen_multi_folder = "./saved_models/mnist_double_pix_multi/"

        base_folder, target_folder = get_newest_folder(dis_sing_folder, cen_sing_folder)
        draw_figure(base_folder, target_folder, ["dis_single", "double_pix_single"])

        base_folder, target_folder = get_newest_folder(dis_multi_folder, cen_multi_folder)
        draw_figure(base_folder, target_folder, ["dis_multi", "double_pix_multi"])


    if mode == "half_attack" or mode == "all":
        cen_sing_folder = "./saved_models/mnist_half_attack_sing/"
        cen_multi_folder = "./saved_models/mnist_half_attack_multi/"

        base_folder, target_folder = get_newest_folder(dis_sing_folder, cen_sing_folder)
        draw_figure(base_folder, target_folder, ["dis_single", "half_attack_single"])

        base_folder, target_folder = get_newest_folder(dis_multi_folder, cen_multi_folder)
        draw_figure(base_folder, target_folder, ["dis_multi", "half_attack_multi"])

    if mode == "defense" or mode == "all":
        cen_sing_folder = "./saved_models/mnist_foolsgold/"
        cen_multi_folder = "./saved_models/mnist_geomedian/"

        base_folder, target_folder = get_newest_folder(dis_sing_folder, cen_sing_folder)
        draw_figure(base_folder, target_folder, ["dis_single", "foolsgold"])

        base_folder, target_folder = get_newest_folder(dis_multi_folder, cen_multi_folder)
        draw_figure(base_folder, target_folder, ["dis_multi", "geomedian"])
    
    if mode == "similarity":
        if not os.path.exists(similarity_figure_folder):
            os.makedirs(similarity_figure_folder)
        
        target_folder = "./saved_models/mnist_dis_multi_simi/"
        base_folder, target_folder = get_newest_folder(dis_sing_folder, target_folder)
        draw_similarity_figure(target_folder, "dis_multi")

        target_folder = "./saved_models/mnist_cen_multi_simi/"
        base_folder, target_folder = get_newest_folder(dis_sing_folder, target_folder)
        draw_similarity_figure(target_folder, "cen_multi")










                            







    


    





