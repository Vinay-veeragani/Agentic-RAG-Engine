import json

from agentic_rag.ingestion.parsed_document import ElementType
from agentic_rag.ingestion.parsers.csv_parser import CsvParser
from agentic_rag.ingestion.parsers.html import HtmlParser
from agentic_rag.ingestion.parsers.json_parser import JsonParser
from agentic_rag.ingestion.parsers.markdown import MarkdownParser
from agentic_rag.ingestion.parsers.text import TextParser


def test_text_parser_splits_paragraphs_on_blank_lines() -> None:
    content = b"First paragraph.\n\nSecond paragraph."
    doc = TextParser().parse(filename="a.txt", content=content)
    assert [e.text for e in doc.elements] == ["First paragraph.", "Second paragraph."]
    assert all(e.element_type == ElementType.PARAGRAPH for e in doc.elements)


def test_markdown_parser_extracts_headings_and_tracks_heading_context() -> None:
    content = b"# Title\n\nIntro paragraph.\n\n## Section\n\nSection paragraph.\n"
    doc = MarkdownParser().parse(filename="a.md", content=content)

    headings = [e for e in doc.elements if e.element_type == ElementType.HEADING]
    paragraphs = [e for e in doc.elements if e.element_type == ElementType.PARAGRAPH]

    assert [h.text for h in headings] == ["Title", "Section"]
    assert paragraphs[0].text == "Intro paragraph." and paragraphs[0].heading == "Title"
    assert paragraphs[1].text == "Section paragraph." and paragraphs[1].heading == "Section"


def test_markdown_parser_extracts_list_items_and_code_blocks() -> None:
    content = b"- item one\n- item two\n\n```python\nprint('hi')\n```\n"
    doc = MarkdownParser().parse(filename="a.md", content=content)

    list_items = [e for e in doc.elements if e.element_type == ElementType.LIST_ITEM]
    code_blocks = [e for e in doc.elements if e.element_type == ElementType.CODE]

    assert [i.text for i in list_items] == ["item one", "item two"]
    assert code_blocks[0].text == "print('hi')"
    assert code_blocks[0].metadata["language"] == "python"


def test_html_parser_extracts_headings_paragraphs_and_tables() -> None:
    content = b"""
    <html><head><title>My Doc</title></head>
    <body>
      <h1>Heading One</h1>
      <p>Paragraph text.</p>
      <table><tr><th>Col A</th><th>Col B</th></tr><tr><td>1</td><td>2</td></tr></table>
    </body></html>
    """
    doc = HtmlParser().parse(filename="a.html", content=content)

    assert doc.title == "My Doc"
    headings = [e for e in doc.elements if e.element_type == ElementType.HEADING]
    paragraphs = [e for e in doc.elements if e.element_type == ElementType.PARAGRAPH]
    tables = [e for e in doc.elements if e.element_type == ElementType.TABLE]

    assert headings[0].text == "Heading One"
    assert paragraphs[0].text == "Paragraph text." and paragraphs[0].heading == "Heading One"
    assert "Col A | Col B" in tables[0].text


def test_csv_parser_renders_rows_as_column_value_pairs() -> None:
    content = b"name,age\nAlice,30\nBob,25\n"
    doc = CsvParser().parse(filename="a.csv", content=content)
    assert [e.text for e in doc.elements] == ["name: Alice, age: 30", "name: Bob, age: 25"]
    assert all(e.element_type == ElementType.TABLE for e in doc.elements)


def test_json_parser_handles_list_of_records() -> None:
    content = json.dumps([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]).encode()
    doc = JsonParser().parse(filename="a.json", content=content)
    assert [e.text for e in doc.elements] == ["name: Alice, age: 30", "name: Bob, age: 25"]


def test_json_parser_handles_arbitrary_object_as_single_element() -> None:
    content = json.dumps({"key": "value", "nested": {"a": 1}}).encode()
    doc = JsonParser().parse(filename="a.json", content=content)
    assert len(doc.elements) == 1
    assert '"key": "value"' in doc.elements[0].text
