#!/usr/bin/env python3
"""Render markdown report to HTML."""

import json
import sys
from pathlib import Path


def markdown_to_html(markdown_text: str) -> str:
    """Convert markdown to HTML with basic styling."""
    # Try using markdown library if available
    try:
        import markdown
        html_content = markdown.markdown(
            markdown_text,
            extensions=['extra', 'codehilite', 'toc']
        )
    except ImportError:
        # Fallback: wrap in pre tags
        html_content = f"<pre style='white-space: pre-wrap; font-family: system-ui;'>{markdown_text}</pre>"
    
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report</title>
    <style>
        body {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.6;
            color: #333;
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }}
        code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin-left: 0;
            padding-left: 20px;
            color: #555;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""


def main():
    # Load the results
    results_path = Path(__file__).parent.parent / "output" / "results.json"
    
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)
    
    with open(results_path) as f:
        data = json.load(f)
    
    markdown_text = data.get("writeup", "")
    
    if not markdown_text:
        print("Error: No writeup found in results.json")
        sys.exit(1)
    
    # Convert to HTML
    html = markdown_to_html(markdown_text)
    
    # Save HTML file
    output_path = results_path.parent / "report.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ HTML report saved to: {output_path}")
    print(f"\nOpen it with: open {output_path}")


if __name__ == "__main__":
    main()

