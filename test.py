import config.config as cfg

with open(cfg.PATH_TO_LABELS, "r") as label_file:

    lines = label_file.readlines()
    labels_dict = dict()

    for i in range(0, len(lines), 4):
        labels_dict[
            int(lines[i + 2].split(":")[1].strip('\n'))] = lines[i + 1].split(":")[1].strip('"\n')


    print(labels_dict)



