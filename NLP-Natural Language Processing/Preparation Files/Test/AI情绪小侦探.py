# AI情绪小侦探

# 1️⃣ 定义情绪词库
positive_words = ["love", "great", "happy", "awesome", "good", "fantastic"]
negative_words = ["bad", "sad", "terrible", "hate", "awful", "angry"]

# 2️⃣ 获取用户输入
sentence = input("请输入一句英文评论: ")

# 3️⃣ 转换为小写
sentence = sentence.lower()

# 4️⃣ 分词
words = sentence.split()

# 5️⃣ 统计情绪
positive_count = 0
negative_count = 0

for word in words:
    if word in positive_words:
        positive_count += 1
    if word in negative_words:
        negative_count += 1

# 6️⃣ 判断情绪
if positive_count > negative_count:
    print("😊 这是一个积极的评论!")
elif negative_count > positive_count:
    print("😢 这是一个消极的评论!")
else:
    print("😐 无法判断情绪")
