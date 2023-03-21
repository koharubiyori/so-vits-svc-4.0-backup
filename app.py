import io
import os
import argparse
import gradio as gr
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
import logging
import json

parser = argparse.ArgumentParser()
parser.add_argument("--user", type=str, help='set gradio user', default=None)
parser.add_argument("--password", type=str, help='set gradio password', default=None)
cmd_opts = parser.parse_args()

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def load_model_func(ckpt_name,cluster_name,config_name):
    global model, cluster_model_path
    
    config_path = "configs/" + config_name

    with open(config_path, 'r') as f:
        config = json.load(f)
    spk_dict = config["spk"]

    ckpt_path = "logs/44k/" + ckpt_name
    cluster_path = "logs/44k/" + cluster_name
    if cluster_name == "no_clu":
            model = Svc(ckpt_path,config_path)
    else:
            model = Svc(ckpt_path,config_path,cluster_model_path=cluster_path)

    spk_list = list(spk_dict.keys())
    return "模型加载成功",gr.Dropdown.update(choices=spk_list)



#read ckpt lsit
file_list = os.listdir("logs/44k")
ckpt_list = []
cluster_list = []
for ck in file_list:
    if os.path.splitext(ck)[-1] == ".pth" and ck[0] != "k":
        ckpt_list.append(ck)
    if ck[0] == "k":
        cluster_list.append(ck)
if not cluster_list:
    cluster_list = ["你没有聚类模型"]
    print("no clu")


def vc_fn(sid, input_audio, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale):
    if input_audio is None:
        return "You need to upload an audio", None
    sampling_rate, audio = input_audio
    # print(audio.shape,sampling_rate)
    duration = audio.shape[0] / sampling_rate
    if duration > 90:
        return "请上传小于90s的音频，需要转换长音频请本地进行转换", None
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    print(audio.shape)
    out_wav_path = "temp.wav"
    soundfile.write(out_wav_path, audio, 16000, format="wav")
    print( cluster_ratio, auto_f0, noise_scale)
    _audio = model.slice_inference(out_wav_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale)
    return "Success", (44100, _audio)


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Basic"):
            gr.Markdown(value="""
                sovits4.0 webui 推理
                
                修改自bilibili@麦哲云

                仅供个人娱乐和非商业用途，禁止用于血腥、暴力、性相关、政治相关内容

                作者：bilibili@羽毛布団（联系请发B站私信）

                """)
            choice_ckpt = gr.Dropdown(label="模型选择", choices=ckpt_list, value="no_model")
            config_choice = gr.Dropdown(label="配置文件", choices=os.listdir("configs"), value="no_config")
            cluster_choice = gr.Dropdown(label="选择聚类模型", choices=cluster_list, value="no_clu")
            loadckpt = gr.Button("加载模型", variant="primary")
            
            sid = gr.Dropdown(label="音色", value="speaker0")
            model_message = gr.Textbox(label="Output Message")
            
            loadckpt.click(load_model_func,[choice_ckpt,cluster_choice,config_choice],[model_message, sid])
            
            gr.Markdown(value="""
                请稍等片刻，模型加载大约需要10秒。后续操作不需要重新加载模型
                """)
            
            
            
            vc_input3 = gr.Audio(label="上传音频（长度小于90秒）")
            vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
            cluster_ratio = gr.Number(label="聚类模型混合比例，0-1之间，默认为0不启用聚类，能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）", value=0)
            auto_f0 = gr.Checkbox(label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声不要勾选此项会究极跑调）", value=False)
            slice_db = gr.Number(label="切片阈值", value=-40)
            noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
            
            vc_submit = gr.Button("转换", variant="primary")
            vc_output1 = gr.Textbox(label="Output Message")
            vc_output2 = gr.Audio(label="Output Audio")
        vc_submit.click(vc_fn, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale], [vc_output1, vc_output2])
        
        app.launch()




