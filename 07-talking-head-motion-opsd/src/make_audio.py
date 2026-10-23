"""Build a small, license-clean speech set with macOS TTS: digital-human style scripts
(ecommerce livestream, corporate training, news, customer service, knowledge explainers).

Writes 16 kHz mono wavs to data/audio/{train,eval,demo}/ and a manifest.
"""
import json
import os
import random
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "audio")

SCRIPTS = [
    # ecommerce
    "家人们，今天这款保温杯真的太值了，三百毫升的容量，二十四小时保温，下单还送杯套。",
    "这件外套是我们店里卖得最好的一款，面料特别舒服，颜色也很显白，大家可以放心拍。",
    "倒计时最后三分钟，库存只剩下两百单，喜欢的朋友抓紧时间，错过今天就要恢复原价了。",
    "很多朋友问这个面霜适不适合敏感肌，我可以告诉大家，它是无香精配方，孕妇也可以用。",
    "我们支持七天无理由退换，运费险也已经帮大家买好了，收到不满意直接退，没有任何风险。",
    "今天直播间的福利是买二送一，再叠加平台满减，算下来每件只要五十九块，真的非常划算。",
    # training / corporate
    "欢迎大家参加本次新员工培训，接下来的一个小时，我会带大家了解公司的组织架构和工作流程。",
    "在处理客户投诉的时候，第一步一定是先倾听，不要急着解释，让客户把情绪表达出来。",
    "数据安全是我们每个人的责任，请不要把公司文件上传到个人网盘，也不要随意点击陌生链接。",
    "本季度我们的销售额同比增长了百分之十八，其中线上渠道的贡献超过了一半。",
    "项目管理的核心是把大目标拆成小任务，每个任务都要有明确的负责人和截止时间。",
    "如果你在报销流程中遇到问题，可以先查看共享文档里的常见问题，再联系财务部同事。",
    # news / announcements
    "各位观众大家好，欢迎收看今天的科技快讯，我们先来关注一下人工智能领域的最新进展。",
    "气象部门预计，未来三天本市将迎来一次明显降温过程，请市民朋友注意添衣保暖。",
    "本次展会共吸引了来自三十多个国家的一千余家企业参展，现场签约金额创下历史新高。",
    "据了解，新的地铁线路将于下个月正式开通运营，全程设置十二座车站，运行时间约四十分钟。",
    # customer service
    "您好，很高兴为您服务，请问您遇到的是登录问题还是支付问题，我来帮您快速处理。",
    "您的订单已经发货了，预计明天下午送达，物流单号我已经通过短信发送到您的手机上。",
    "非常抱歉给您带来不便，我们已经为您申请了优先处理，工作人员会在两小时内联系您。",
    "如果您需要修改收货地址，可以在订单详情页点击修改，在发货之前都是可以免费更改的。",
    # knowledge explainers
    "很多人以为喝咖啡会脱水，其实适量饮用并不会，咖啡里的水分基本可以抵消它的利尿作用。",
    "复利的意思是利息也会产生利息，时间越长，这种滚雪球的效果就越明显，所以理财要趁早。",
    "睡前半小时尽量不要看手机，屏幕的蓝光会抑制褪黑素分泌，让人更难入睡。",
    "学习一门新语言，最重要的不是背单词，而是每天坚持开口说，哪怕只有十分钟。",
    "为什么天空是蓝色的？因为阳光穿过大气层时，波长较短的蓝光更容易被散射到各个方向。",
    "跑步前一定要做五到十分钟的热身，可以有效降低膝盖和脚踝受伤的风险。",
    "番茄工作法就是专注工作二十五分钟，然后休息五分钟，每完成四轮再休息一次长的。",
    "信用卡的账单日和还款日是两个不同的概念，搞清楚它们可以帮你多获得一些免息期。",
    "其实植物也需要休息，夏天中午最好不要浇水，温差太大容易伤到根系。",
    "我们常说的五险一金，指的是养老、医疗、失业、工伤、生育保险，再加上住房公积金。",
    "如果手机电池不耐用了，可以先检查一下后台有没有一直在运行的应用。",
    "做饭的时候先放盐还是后放盐，其实会影响菜的口感，炒青菜最好出锅前再放。",
    "今天我想和大家聊一聊时间管理，很多时候我们不是没有时间，而是没有想清楚什么最重要。",
    "说到健康饮食，大家可以记住一个简单的原则，每顿饭有一半是蔬菜，其余是主食和蛋白质。",
    "很高兴今天能和大家分享我们团队过去一年的成长，也想借这个机会感谢每一位伙伴的付出。",
    "接下来我会用三个例子来说明这个问题，希望大家听完之后，能有一些新的思考。",
]
VOICES = ["Tingting", "Eddy (中文（中国大陆）)", "Flo (中文（中国大陆）)", "Reed (中文（中国大陆）)",
          "Sandy (中文（中国大陆）)", "Shelley (中文（中国大陆）)", "Rocko (中文（中国大陆）)"]


def synth(text, voice, rate, wav):
    aiff = wav.replace(".wav", ".aiff")
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", aiff, text], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", aiff, "-ac", "1", "-ar", "16000", wav], check=True)
    os.remove(aiff)


def main():
    rng = random.Random(0)
    idx = list(range(len(SCRIPTS)))
    rng.shuffle(idx)
    splits = {"demo": idx[:3], "eval": idx[3:11], "train": idx[11:]}
    manifest = []
    for split, ids in splits.items():
        os.makedirs(os.path.join(OUT, split), exist_ok=True)
        for k, i in enumerate(ids):
            voice = VOICES[(i + k) % len(VOICES)]
            rate = rng.choice([170, 185, 200, 215])
            wav = os.path.join(OUT, split, f"{split}_{k:02d}.wav")
            synth(SCRIPTS[i], voice, rate, wav)
            manifest.append({"split": split, "path": os.path.relpath(wav, ROOT), "voice": voice, "rate": rate,
                             "text": SCRIPTS[i]})
    with open(os.path.join(ROOT, "data", "audio_manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    for s in splits:
        print(s, len(splits[s]))


if __name__ == "__main__":
    main()
