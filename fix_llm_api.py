filepath = r"E:\Agent论文\perfect\src\llm_api.py"

with open(filepath, "rb") as f:
    content = f.read()

# 修复：把 'gpt5.1' 改回 'gpt5.1'（字典key名）
content = content.replace(b"'gpt5.1': {", b"'gpt5.1': {")

# 验证能否UTF-8解码
try:
    text = content.decode("utf-8")
    print("UTF-8 decode: OK")
except Exception as e:
    print(f"UTF-8 decode failed: {e}")
    # 尝试修复
    text = content.decode("utf-8", errors="replace")
    content = text.encode("utf-8")

# 保存
with open(filepath, "wb") as f:
    f.write(content)

# 验证结果
with open(filepath, "rb") as f:
    fixed = f.read()

idx = fixed.find(b"'gpt5.1':")
if idx >= 0:
    print("Found gpt5 config at byte", idx)
    print(fixed[idx:idx+150])
else:
    print("WARNING: gpt5 not found!")

print("Done!")