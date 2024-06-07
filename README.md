# Classification
结合HanLP语义分析和腾讯text smartSDK的语言分类工具
# 使用方法
****
## 1. 安装HanLP
```bash
pip install hanlp
```

## 2. 下载腾讯texSmart SDK
```bash
https://ai.tencent.com/ailab/nlp/texsmart/zh/download.html
```
**建议使用中型 (M)模型**

## 3. 运行
### 复制项目至腾讯TexSmart SDK的根目录
> 项目文件有三个**app.py, config.json，keywords.txt**
### Windows
在TexSmart SDK的目录下，按住**shift+右键**“在命令行中打开”
```bash
python app.py
```
### Linux
```bash
python3 app.py
```

## 4. 输入/输出样例

#### config.json 里面填入分类
```bash
{
    "科技": "computer，network，artificial intelligence",
    "水果": "苹果,桃子,李子,梨子,柿子",
    "奋斗": "每天的第一缕阳光是对昨日辛勤的肯定，也是今日拼搏的号角。 疲惫时，请记得，每一滴汗水都是浇灌成功的雨露",
    "文艺": "最后，我决定把你放在一群普通的朋友中，不要勉强和怀旧在别人伤害你之前，你必须像个绅士一样生活明明知道这辈子都不可能再跟你有关联，可还是忍不住想你，连睡觉都会经常梦见你，我讨厌这样的我，我想像你对我一样绝情可我做不到。我还是那么卑微。还是那么在乎你。",
    "古文": "豫章故郡，洪都新府。星分翼轸，地接衡庐。襟三江而带五湖，控蛮荆而引瓯越。物华天宝，龙光射牛斗之墟；人杰地灵，徐孺下陈蕃之榻。雄州雾列，俊采星驰。台隍枕夷夏之交，宾主尽东南之美。都督阎公之雅望，棨戟遥临；宇文新州之懿范，襜帷暂驻。十旬休假，胜友如云；千里逢迎，高朋满座。腾蛟起凤，孟学士之词宗；紫电青霜，王将军之武库。家君作宰，路出名区；童子何知，躬逢胜饯"
    # 可以自行添加删除
}
```

#### keywords.txt 中存放需要分类的单词/句子，每行一个
```bash
橙子
香蕉
葡萄
西瓜
草莓
电脑
blockchain
robotics
quantum computing
virtual reality
nanotechnology
遥襟甫畅，逸兴遄飞。爽籁发而清风生，纤歌凝而白云遏。睢园绿竹，气凌彭泽之樽；邺水朱华，光照临川之笔。四美具，二难并。穷睇眄于中天，极娱游于暇日。天高地迥，觉宇宙之无穷；
我们其实性格很不合。他早就发现了，我也是。我们只是……舍不得分开
你在做什么？我在仰望天空。30度的仰望是什么？是我想念她的角度。为什么要把头抬到30度？为了不让我的眼泪掉下来？
对乐于苦斗的人来说，苦斗不是憾事，而是乐事。” ——托马斯
```

### 输出会自动生成一个analysis.txt
```bash
橙子: 水果【实体识别】
香蕉: 水果【实体识别】
葡萄: 水果【实体识别】
西瓜: 水果【实体识别】
草莓: 水果【实体识别】
电脑: 科技【实体识别】
blockchain: 科技【实体识别】
robotics: 科技【实体识别】
quantum computing: 科技【实体识别】
virtual reality: 科技【实体识别】
nanotechnology: other疑似【科技, 0.45】
遥襟甫畅，逸兴遄飞。爽籁发而清风生，纤歌凝而白云遏。睢园绿竹，气凌彭泽之樽；邺水朱华，光照临川之笔。四美具，二难并。穷睇眄于中天，极娱游于暇日。天高地迥，觉宇宙之无穷；: 古文【字符匹配】
我们其实性格很不合。他早就发现了，我也是。我们只是……舍不得分开: 文艺【字符匹配】
你在做什么？我在仰望天空。30度的仰望是什么？是我想念她的角度。为什么要把头抬到30度？为了不让我的眼泪掉下来？: 文艺【字符匹配】
对乐于苦斗的人来说，苦斗不是憾事，而是乐事。” ——托马斯: 奋斗【字符匹配】
```

### 日志文件会写入在log.txt里
```bash
分类的实体识别结果:
科技: product.device, kafield.technology
水果: food.fruit,
...
处理词条: 橙子
识别到的实体: food.fruit
...
分类的相关内容:
科技: GPS, radar, radio, VHF, compass, autopilot,
...
处理词条: 橙子
识别到的实体: food.fruit
...
分类处理时间: 2024-06-07 15:29:02.442467
分类结果统计:
科技: 5 个词条
水果: 5 个词条
奋斗: 1 个词条
文艺: 2 个词条
古文: 1 个词条
疑似分类统计:
疑似 科技: 1 个词条
分类处理时间: 2024-06-07 15:29:02.442467
分类结果统计:
科技: 5 个词条
水果: 5 个词条
奋斗: 1 个词条
文艺: 2 个词条
古文: 1 个词条
疑似分类统计:
疑似 科技: 1 个词条
疑似 水果: 0 个词条
疑似 奋斗: 0 个词条
疑似 文艺: 0 个词条
疑似 古文: 0 个词条

```

## 5. 注意事项
**请注意，如果analysis.txt已存在，程序运行时会读取该文件，并且跳过analysis.txt中已经分析成功的词条**
