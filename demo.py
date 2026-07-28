"""
AI 顾客意向识别 + 跟进建议生成 —— 最小原型
输入：5 段微信导购-顾客模拟聊天
输出：每段的意向判断 + 1 条跟进建议
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────
API_KEY = "这里配置您的apikey即可"
BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

# ── 模拟聊天数据 ──────────────────────────────────────
CONVERSATIONS = [
    {
        "id": "会话 1 | 顾客：林",
        "participants": "导购：小美",
        "history": """林    10:32  在吗
林    10:32  你们朋友圈那件米白色的针织开衫还有吗[图片]
小美  10:40  在的亲～ 那款是我们这周新到的，米白还有货哦
林    10:41  我158、92斤  拍S还是M
小美  10:43  您这身材建议S，版型本身偏宽松，S穿正好显瘦
林    10:44  好  那件多少钱来着
小美  10:44  319，这周会员日刚好满300减30
林    10:45  行  我要一件S的
林    10:45  怎么付  直接微信转你吗"""
    },
    {
        "id": "会话 2 | 顾客：Cindy",
        "participants": "导购：阿凯",
        "history": """（以下为 3 天前的对话）
Cindy 19:20  这条阔腿裤多少钱
阿凯  19:25  您好～ 这条是高腰显瘦的，原价459，现在活动价399
Cindy 19:26  哦  有点小贵
阿凯  19:27  这个是醋酸面料亲，垂感特别好不起球，很多老顾客回购的
Cindy 19:30  嗯  我再看看
阿凯  19:31  好的亲～ 有需要随时找我哈，我先帮您留一条
（此后 3 天，Cindy 无任何消息）"""
    },
    {
        "id": "会话 3 | 顾客：王姐",
        "participants": "导购：小美",
        "history": """小美  09:00  [朋友圈] 早安～ 今日穿搭：秋天的第一件风衣🍂[图片]
王姐  09:15  这个拍得好看
小美  09:16  谢谢王姐～ 这是上周新到的风衣。您上次买的那件大衣穿着还合适吧？
王姐  09:20  挺好的  就是最近忙  没怎么出门
小美  09:21  哈哈 那您注意身体哈～ 天凉了记得多穿点
王姐  09:22  嗯嗯  你也是"""
    },
    {
        "id": "会话 4 | 顾客：圆圆",
        "participants": "导购：阿凯",
        "history": """圆圆 14:02  你好  我前天收到的那件卫衣
圆圆 14:02  袖口这里有个线头  还开了一点缝[图片]
圆圆 14:03  我才穿了一次
阿凯  14:30  亲实在抱歉～ 您方便拍一下整体和细节我看一下嘛
圆圆 14:31  喏  就这样[图片]
圆圆 14:33  我是退还是换啊  有点闹心"""
    },
    {
        "id": "会话 5 | 顾客：娜娜（常客）",
        "participants": "导购：小美",
        "history": """娜娜 21:10  最近有上新吗
娜娜 21:10  上次那个羊毛衫我穿着特别舒服  还想再入几件
小美  21:40  有的娜娜～ 这两天刚到一批秋冬新款，羊毛的有好几个颜色[图片][图片]
娜娜 21:42  哇  这个豆沙色好看
娜娜 21:43  不过我这周要出差  下周回来再说哈
小美  21:44  好嘞～ 那我先帮您把豆沙色这件留着，下周回来我招呼您"""
    },
]

# ── Prompt 设计 ──────────────────────────────────────
SYSTEM_PROMPT = """你是一家女装连锁门店的「AI 跟单助手」。你的工作是从导购与顾客的微信聊天记录中，快速识别顾客当前的购买意向，并给出 1 条最该马上去做的跟进建议。

## 你的判断维度

### 意向判断（三类）：
- **高意向**：顾客已明确表达购买意愿（要了款式、谈到了付款方式），或正在处理售后且情绪平稳
- **中意向**：顾客表现出兴趣但还在犹豫（问了价格、对比款式），或对话中断在购买决策前
- **低意向**：仅为社交互动（点赞朋友圈、寒暄），无明确购买信号

### 跟进建议原则：
- 每段对话只给 1 条建议，必须具体、可执行（不要空话）
- 考虑对话的时效性——3 天没回复 vs 刚聊完，策略不同
- 高意向优先促成成交动作（发付款码/确认收货地址）；中意向降低决策门槛（发搭配图/限时优惠）；低意向先建立联系（不推货，找话题）
- 建议要符合导购身份语气，自然不僵硬，像真人同事在提醒你

## 输出格式（严格 JSON）
```json
{
  "conversations": [
    {
      "id": "会话 X",
      "intent": "高意向 / 中意向 / 低意向",
      "intent_reason": "20 字以内说清判断依据",
      "suggestion": "给导购的具体跟进建议，一句话"
    }
  ]
}
```

请严格按此 JSON 格式输出，不要包含其他内容。"""


def build_user_prompt(conversations):
    """将所有聊天组装成一条 user prompt"""
    blocks = []
    for conv in conversations:
        blocks.append(f"【{conv['id']}】（{conv['participants']}）\n{conv['history']}")
    return "\n\n---\n\n".join(blocks) + "\n\n请分析以上每一段对话，输出 JSON。"


def main():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print("=" * 60)
    print("AI 顾客意向识别 + 跟进建议生成")
    print("=" * 60)

    payload = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(CONVERSATIONS)},
    ]

    print(f"\n正在分析 {len(CONVERSATIONS)} 段聊天记录……\n")

    resp = client.chat.completions.create(
        model=MODEL,
        messages=payload,
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    # ── 输出结果 ────────────────────────────────────
    import json

    result = json.loads(resp.choices[0].message.content)
    for c in result["conversations"]:
        print(f"\n{'─' * 50}")
        print(f"  {c['id']}")
        print(f"  [意向] {c['intent']}（{c['intent_reason']}）")
        print(f"  [建议] {c['suggestion']}")
        print(f"{'─' * 50}")
        print()

    print("[完成] 分析完成")


if __name__ == "__main__":
    main()
