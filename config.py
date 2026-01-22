import os

class Config:

    num_folds = 5
    select_metric = "mean"


    device = "cuda"
    seed   = 666


    data_root = r"./data/NUS_dataset"
    camera_sensitivity_path = r"./data/Canon_1D_Mark_III.mat"


    max_epochs   = 1000
    batch_size   = 256
    num_workers  = 4
    lr           = 1e-4
    weight_decay = 0.0
    use_amp      = True


    patch_size     = 8
    spectral_bands = 31
    spec_feat_k    = 4


    wp_quantile        = 0.95
    bright_topk_frac   = 0.10
    use_pca_bright_train = True
    use_pca_bright_eval  = True
    pb_mode = "bright_dark"


    kmeans_unit_normalize = True
    kmeans_conf_weight    = True
    conf_brightness_power = 0.5
    kmeans_threads        = 1


    alpha_viz_topN_images = 3
    fig_dir   = "./figs"
    save_dir  = "./checkpoints"
    best_model_name = "best_mean_notau_3.pth"
    profiling_csv   = "./logs/profile.csv"
    analysis_csv    = "./logs/analysis.csv"
    log_profile     = False


os.makedirs(Config.save_dir, exist_ok=True)
os.makedirs(os.path.dirname(Config.profiling_csv), exist_ok=True)
os.makedirs(Config.fig_dir, exist_ok=True)
