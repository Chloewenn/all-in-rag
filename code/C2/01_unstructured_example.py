from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

# from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured加载并解析PDF文档
elements = partition_pdf(
    filename=pdf_path,
    content_type="application/pdf",
    strategy='ocr_only',
    languages=["chi_sim", "eng"], # 简体中文 + 英文
)
# strategy: ['hi_res', 'ocr_only'], hi_res 既关注“写了什么”，也关注“内容在页面上如何组织”；ocr_only 主要关注“图片里有哪些文字”
# 'hi_res' 希望区分元素类型、分析复杂排版、提取表格结构
# 'ocr_only' 主要需要读取扫描件中的文字，不需要专门识别版面结构

# 输出文件放在脚本所在目录；不受终端当前工作目录影响。
output_path = Path(__file__).resolve().with_name("01_output.txt")

# "w" 表示每次运行覆盖文件；encoding="utf-8" 用于保存中文。
# redirect_stdout 将这个代码块内的 print 输出写入文件，无需逐个修改 print。
# 这里只保存解析结果；上面 partition() 产生的下载进度、警告等仍在终端显示。
with output_path.open("w", encoding="utf-8") as output_file, redirect_stdout(output_file):
    # 打印解析结果
    print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

    # 统计元素类型
    types = Counter(e.category for e in elements)
    print(f"元素类型: {dict(types)}")

    # 显示所有元素
    print("\n所有元素:")
    for i, element in enumerate(elements, 1):
        print(f"Element {i} ({element.category}):")
        print(element)
        print("=" * 60)

# 离开 with 后恢复终端输出，文件也已关闭；再显示保存的内容与文件位置。
print(output_path.read_text(encoding="utf-8"), end="")
print(f"\n输出已保存到: {output_path}")
