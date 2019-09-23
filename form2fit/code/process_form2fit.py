import glob
import os
import pickle


if __name__ == "__main__":
    dump_dir = "./dump/"
    form2fit_pkls = glob.glob(dump_dir + "/*")
    kit_poses = {}
    for pkl in form2fit_pkls:
        if os.path.basename(pkl) == "ORB_PE_poses.pkl":
            continue
        kit_poses[os.path.basename(pkl).split("_")[0]] = pickle.load(open(pkl, "rb"))
    with open(os.path.join(dump_dir, "form2fit_poses.pkl"), "wb") as fp:
        pickle.dump(kit_poses, fp)