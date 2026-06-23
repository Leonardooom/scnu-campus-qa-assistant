import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests
from lxml import html
from pypdf import PdfReader


TODAY = date.today().isoformat()
OUTPUT_DIR = Path("generated_scnu_rag")
RAW_DIR = OUTPUT_DIR / "raw"
DOCS_DIR = OUTPUT_DIR / "docs"


SOURCES = [
    {
        "slug": "jw_site_map",
        "title_hint": "华南师范大学本科生院网站地图与业务栏目",
        "category": "教务",
        "source_url": "https://jw.scnu.edu.cn/wzdt/",
        "is_time_sensitive": True,
        "keywords": ["教务", "成绩", "选课", "课表", "学籍", "考试", "毕业", "学位"],
    },
    {
        "slug": "jw_department_contacts",
        "title_hint": "本科生院科室介绍与联系方式",
        "category": "教务",
        "source_url": "https://jw.scnu.edu.cn/about/dept/",
        "is_time_sensitive": True,
        "keywords": ["本科生院", "教务处", "成绩管理", "学籍管理", "排课", "排考", "联系方式"],
    },
    {
        "slug": "jw_regulations",
        "title_hint": "本科生院规章制度汇总",
        "category": "教务",
        "source_url": "https://jw.scnu.edu.cn/guizhangzhidu/",
        "is_time_sensitive": True,
        "keywords": ["学籍管理", "成绩管理", "转专业", "辅修", "毕业", "学位", "缓考", "考试"],
    },
    {
        "slug": "lib_faq",
        "title_hint": "图书馆常见问题解答",
        "category": "图书馆",
        "source_url": "https://lib.scnu.edu.cn/guide/changjianwentijieda/",
        "is_time_sensitive": True,
        "keywords": ["图书馆", "续借", "借阅", "逾期", "借书", "我的图书馆", "密码"],
    },
    {
        "slug": "lib_reader_rules",
        "title_hint": "图书馆读者管理与服务规定",
        "category": "图书馆",
        "source_url": "https://lib.scnu.edu.cn/guide/duzheguanliyufuwu/",
        "is_time_sensitive": True,
        "keywords": ["入馆", "借阅", "校园卡", "开放时间", "预约", "服务台"],
    },
    {
        "slug": "lib_borrow_card",
        "title_hint": "图书馆借阅证管理办法",
        "category": "图书馆",
        "source_url": "https://lib.scnu.edu.cn/guide/jieyuezhenguanlibanfa/",
        "is_time_sensitive": True,
        "keywords": ["借阅证", "借阅权限", "续借次数", "预约", "委托", "全日制学生"],
    },
    {
        "slug": "lib_cross_campus",
        "title_hint": "图书馆预约与跨校园委托借阅",
        "category": "图书馆",
        "source_url": "https://lib.scnu.edu.cn/a/20240709/2502.html",
        "is_time_sensitive": True,
        "keywords": ["图书馆", "预约", "跨校园", "委托借阅", "馆藏查询", "微信公众号"],
    },
    {
        "slug": "nc_home",
        "title_hint": "网络信息中心综合信息服务",
        "category": "校园卡与网络",
        "source_url": "https://nc.scnu.edu.cn/",
        "is_time_sensitive": True,
        "keywords": ["网络信息中心", "校园网", "师生服务中心", "一站式办事", "正版化服务"],
    },
    {
        "slug": "sso_login",
        "title_hint": "统一身份认证登录说明",
        "category": "校园卡与网络",
        "source_url": "https://sso.scnu.edu.cn/",
        "is_time_sensitive": True,
        "keywords": ["统一身份认证", "账号", "密码", "学号", "一卡通号", "初始密码"],
        "manual_content": (
            "华南师范大学统一身份认证是访问校内信息系统的统一登录入口。"
            "师生办理校园网、办事平台、教务和部分图书馆及自助服务时，通常需要先通过统一身份认证登录。"
            "账号一般与学号、工号或一卡通号关联，初始密码和身份信息规则以官方登录页说明为准。"
        ),
    },
    {
        "slug": "nc_campus_card_guide",
        "title_hint": "校园卡使用指南",
        "category": "校园卡与网络",
        "source_url": "https://nc.scnu.edu.cn/a/20140421/168.html",
        "is_time_sensitive": True,
        "keywords": ["校园卡", "挂失", "解挂", "补办", "注销", "一卡通服务大厅"],
    },
    {
        "slug": "nc_card_service",
        "title_hint": "一卡通服务说明",
        "category": "校园卡与网络",
        "source_url": "https://nc.scnu.edu.cn/a/20190709/342.html",
        "is_time_sensitive": True,
        "keywords": ["一卡通", "校园卡", "充值", "微信充值", "三校区", "服务电话"],
    },
    {
        "slug": "nc_identity_notice",
        "title_hint": "电子身份注册中心与一卡通申请流程调整",
        "category": "校园卡与网络",
        "source_url": "https://nc.scnu.edu.cn/a/20220909/663.html",
        "is_time_sensitive": True,
        "keywords": ["电子身份", "一卡通", "门禁", "校园网", "交换生", "注册中心"],
    },
    {
        "slug": "nc_self_print",
        "title_hint": "自助打印服务",
        "category": "教务",
        "source_url": "https://nc.scnu.edu.cn/a/20200330/387.html",
        "is_time_sensitive": True,
        "keywords": ["自助打印", "成绩单", "在读证明", "课程修读证明", "统一身份认证"],
    },
    {
        "slug": "nc_wifi_pdf",
        "title_hint": "校园无线网 SCNUNET 使用说明",
        "category": "校园卡与网络",
        "source_url": "https://statics.scnu.edu.cn/pics/nc/2024/0308/1709861622262756.pdf",
        "is_time_sensitive": True,
        "keywords": ["SCNUNET", "校园无线网", "wifi.scnu.edu.cn", "无感认证", "统一身份认证"],
    },
    {
        "slug": "hqc_home",
        "title_hint": "后勤管理处服务电话与服务栏目",
        "category": "宿舍与后勤",
        "source_url": "https://hqc.scnu.edu.cn/",
        "is_time_sensitive": True,
        "keywords": ["后勤", "报修", "宿管中心", "物业中心", "服务电话", "石牌", "大学城"],
    },
    {
        "slug": "swc_repair",
        "title_hint": "汕尾校区校园设施设备维修服务",
        "category": "宿舍与后勤",
        "source_url": "https://swc.scnu.edu.cn/a/20250227/50.html",
        "is_time_sensitive": True,
        "keywords": ["报修", "维修", "宿舍", "空调", "门锁", "24小时", "响应时间"],
    },
    {
        "slug": "career_home",
        "title_hint": "就业创业指导中心首页与联系方式",
        "category": "就业",
        "source_url": "https://career.scnu.edu.cn/",
        "is_time_sensitive": True,
        "keywords": ["就业", "就业指导中心", "实习", "宣讲会", "双选会", "联系方式"],
    },
    {
        "slug": "career_student_service",
        "title_hint": "就业学生服务栏目",
        "category": "就业",
        "source_url": "https://career.scnu.edu.cn/module/news?type_id=20456",
        "is_time_sensitive": True,
        "keywords": ["就业手续", "毕业生", "学生服务", "办理指导手册", "就业平台"],
    },
    {
        "slug": "student_aid_intro_pdf",
        "title_hint": "华南师范大学学生资助工作简介",
        "category": "奖助与学生事务",
        "source_url": "https://statics.scnu.edu.cn/pics//2017/1023/1508746326748451.pdf",
        "is_time_sensitive": True,
        "keywords": ["学生资助", "国家奖学金", "国家助学金", "勤工助学", "临时困难补助", "绿色通道"],
    },
    {
        "slug": "student_affairs_scholarship_tag",
        "title_hint": "学生工作部奖学金相关通知标签页",
        "category": "奖助与学生事务",
        "source_url": "https://module.scnu.edu.cn/tag-3358-%E5%A5%96%E5%AD%A6%E9%87%91-11-74.html",
        "is_time_sensitive": True,
        "keywords": ["奖学金", "学生工作部", "通知公告", "国家奖学金", "国家励志奖学金", "评选"],
    },
    {
        "slug": "student_aid_tag",
        "title_hint": "学生工作部资助相关标签页",
        "category": "奖助与学生事务",
        "source_url": "https://module.scnu.edu.cn/tag-3358-%E8%B5%84%E5%8A%A9-1-74.html",
        "is_time_sensitive": True,
        "keywords": ["资助", "助学贷款", "家庭经济困难", "学生资助政策", "绿色通道"],
    },
    {
        "slug": "psychology_service_contacts",
        "title_hint": "汕尾校区心理健康教育与咨询中心联系方式",
        "category": "心理",
        "source_url": "https://swc.scnu.edu.cn/organization/dangzhengzhinengbumen/2025/0904/2.html",
        "is_time_sensitive": True,
        "keywords": ["心理咨询", "心理健康教育与咨询中心", "学生事务管理办公室", "汕尾校区", "联系方式"],
    },
]


DROP_PATTERNS = [
    "Copyright",
    "华南师范大学 版权所有",
    "All Rights Reserved",
    "当前位置",
    "返回首页",
    "Image",
    "可信网站验证",
    "诚信网站验证",
    "安全联盟",
]


SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        )
    }
)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(token in line for token in DROP_PATTERNS):
            continue
        if line in {"首页", "主页", "新闻", "服务指南", "联系我们", "Chatbot"}:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def fetch_binary(url: str) -> bytes:
    response = SESSION.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def extract_pdf_text(blob: bytes) -> str:
    temp_path = RAW_DIR / "_temp_pdf_extract.pdf"
    temp_path.write_bytes(blob)
    try:
        reader = PdfReader(str(temp_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return normalize_text("\n".join(pages))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def score_xpath(doc: html.HtmlElement, xpath_expr: str) -> tuple[int, str]:
    nodes = doc.xpath(xpath_expr)
    texts = []
    for node in nodes:
        if isinstance(node, str):
            value = node.strip()
        else:
            value = " ".join(node.xpath(".//text()"))
        value = normalize_text(value)
        if value:
            texts.append(value)
    joined = "\n\n".join(texts)
    return len(joined), joined


def extract_html_text(body: bytes) -> tuple[str, str]:
    doc = html.fromstring(body)
    for bad in doc.xpath("//script|//style|//noscript|//header|//footer|//nav"):
        parent = bad.getparent()
        if parent is not None:
            parent.remove(bad)

    title = ""
    title_nodes = doc.xpath("//h1/text()")
    if title_nodes:
        title = normalize_text(" ".join(title_nodes))
    if not title:
        title_tag = doc.xpath("//title/text()")
        title = normalize_text(" ".join(title_tag))

    candidates = [
        "//article",
        "//main",
        "//div[contains(@class,'article')]",
        "//div[contains(@class,'content')]",
        "//div[contains(@class,'detail')]",
        "//div[contains(@class,'txt')]",
        "//div[contains(@class,'post')]",
        "//section",
        "//body",
    ]
    best_text = ""
    best_score = -1
    for expr in candidates:
        score, text = score_xpath(doc, expr)
        if score > best_score:
            best_score = score
            best_text = text
    return title, best_text


def fetch_document(source: dict) -> dict:
    url = source["source_url"]
    blob = fetch_binary(url)
    suffix = ".pdf" if url.lower().endswith(".pdf") else ".html"
    raw_path = RAW_DIR / f"{source['slug']}{suffix}"
    raw_path.write_bytes(blob)

    if suffix == ".pdf":
        title = source["title_hint"]
        content = extract_pdf_text(blob)
    else:
        title, content = extract_html_text(blob)
        if not title:
            title = source["title_hint"]
    title = title or source["title_hint"]
    content = normalize_text(content)
    if len(content) < 120 and source.get("manual_content"):
        content = normalize_text(source["manual_content"])

    return {
        "title": title,
        "category": source["category"],
        "source_url": url,
        "source_site": urlparse(url).netloc,
        "last_checked": TODAY,
        "is_time_sensitive": source["is_time_sensitive"],
        "keywords": source["keywords"],
        "content": content,
        "slug": source["slug"],
    }


def write_markdown(doc: dict) -> None:
    md_path = DOCS_DIR / f"{doc['slug']}.md"
    lines = [
        f"# {doc['title']}",
        "",
        f"- category: {doc['category']}",
        f"- source_url: {doc['source_url']}",
        f"- source_site: {doc['source_site']}",
        f"- last_checked: {doc['last_checked']}",
        f"- is_time_sensitive: {str(doc['is_time_sensitive']).lower()}",
        f"- keywords: {', '.join(doc['keywords'])}",
        "",
        "## content",
        "",
        doc["content"],
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    docs = []
    errors = []
    for source in SOURCES:
        try:
            doc = fetch_document(source)
            if len(doc["content"]) < 120 and not source.get("manual_content"):
                raise ValueError("content too short after extraction")
            docs.append(doc)
            write_markdown(doc)
            print(f"OK  {source['slug']}  {len(doc['content'])} chars")
        except Exception as exc:
            errors.append(
                {
                    "slug": source["slug"],
                    "source_url": source["source_url"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"ERR {source['slug']}  {exc}")

    docs = sorted(docs, key=lambda item: (item["category"], item["slug"]))
    sources_manifest = [
        {
            "slug": item["slug"],
            "title": item["title"],
            "category": item["category"],
            "source_url": item["source_url"],
            "source_site": item["source_site"],
            "last_checked": item["last_checked"],
            "is_time_sensitive": item["is_time_sensitive"],
            "keywords": item["keywords"],
            "content_length": len(item["content"]),
        }
        for item in docs
    ]

    (OUTPUT_DIR / "scnu_rag_corpus.json").write_text(
        json.dumps(docs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "scnu_rag_corpus.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in docs),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "scnu_rag_sources.json").write_text(
        json.dumps(
            {
                "generated_on": TODAY,
                "count": len(docs),
                "documents": sources_manifest,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(docs)} documents, {len(errors)} errors.")


if __name__ == "__main__":
    main()
