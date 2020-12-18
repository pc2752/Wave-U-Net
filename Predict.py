from sacred import Experiment
from Config import config_ingredient
import Evaluate
import os

ex = Experiment('Waveunet Prediction', ingredients=[config_ingredient])

@ex.config
def cfg():
    model_path = os.path.join('../../Darius/Wave-U-Net/Source_Estimates/unet_968405-112000_csd',"968405-112000") # Load stereo vocal model by default
    input_path = os.path.join("../../Darius/Wave-U-Net/test_set_mixes","dcs","DCS_TPFullChoir_mix.wav") # Which audio file to separate
    output_path = './' # Where to save results. Default: Same location as input.

@ex.automain
def main(cfg, model_path, input_path, output_path):
    model_config = cfg["model_config"]
    Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)


# Others
# CH01_Bach_audio.wav
# CH04_Mendelssohn_audio.wav
# CH05_Byrd_audio.wav

# unet_l1_283877-228000-noESMUC
# waveunet_755824-246000-noESMUC