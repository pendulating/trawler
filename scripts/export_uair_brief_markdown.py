#!/usr/bin/env python3
"""
Convert the UAIR brief LaTeX manuscript into Markdown using pandoc.

The script will download a pandoc binary via pypandoc on first run if one is
not already available on the system, then execute the conversion with sensible
defaults for the repository layout.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUSCRIPT_DIR = PROJECT_ROOT / "papers" / "uair_brief"
DEFAULT_MAIN_FILE = DEFAULT_MANUSCRIPT_DIR / "MAIN.tex"
DEFAULT_OUTPUT = DEFAULT_MANUSCRIPT_DIR / "uair_brief.md"
DEFAULT_PUBLIC_DIR = DEFAULT_MANUSCRIPT_DIR / "public" / "assets" / "uair-brief"
DEFAULT_RESOURCE_DIRS = ("sections", "sections/prompts", "figures", "tables", "Images")
DEFAULT_ASSET_DIRS = ("Images", "figures", "tables")


class PandocUnavailableError(RuntimeError):
    """Raised when pandoc cannot be provided."""


def expand_latex_inputs(tex_file: Path, base_dir: Path, processed: set[str] | None = None) -> str:
    """
    Recursively expand \\input commands in LaTeX files.
    """
    if processed is None:
        processed = set()
    
    # Avoid infinite recursion
    tex_path = tex_file.resolve()
    if str(tex_path) in processed:
        return ""
    processed.add(str(tex_path))
    
    content = tex_file.read_text(encoding="utf-8")
    
    # Pattern to match \input{filename} or \input filename
    input_pattern = r'\\input\{([^}]+)\}|\\input\s+([^\s}]+)'
    
    def replace_input(match):
        filename = match.group(1) or match.group(2)
        # Remove .tex extension if present
        if filename.endswith('.tex'):
            filename = filename[:-4]
        
        # Try relative to base_dir first (most LaTeX inputs are relative to document root)
        input_file = base_dir / filename
        if not input_file.exists():
            # Try with .tex extension
            input_file = base_dir / f"{filename}.tex"
        
        # If not found, try relative to current file's directory
        if not input_file.exists():
            current_dir = tex_file.parent
            input_file = current_dir / filename
            if not input_file.exists():
                input_file = current_dir / f"{filename}.tex"
        
        if input_file.exists():
            return expand_latex_inputs(input_file, base_dir, processed)
        else:
            # If file not found, return the original command
            return match.group(0)
    
    # Replace all \input commands
    expanded = re.sub(input_pattern, replace_input, content)
    
    return expanded


def ensure_pandoc() -> List[str]:
    """
    Locate a pandoc binary or download it via pypandoc as a fallback.

    Returns the command prefix required to invoke pandoc.
    """
    if shutil.which("pandoc"):
        return ["pandoc"]

    try:
        import pypandoc  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise PandocUnavailableError(
            "pandoc is not installed and pypandoc is unavailable. "
            "Install it with `uv pip install pypandoc` and rerun the script."
        ) from exc

    try:
        pandoc_path = Path(pypandoc.get_pandoc_path())  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        pypandoc.download_pandoc()  # type: ignore[attr-defined]
        pandoc_path = Path(pypandoc.get_pandoc_path())  # type: ignore[attr-defined]

    if not pandoc_path.exists():
        raise PandocUnavailableError(
            f"Resolved pandoc path {pandoc_path} does not exist."
        )

    return [str(pandoc_path)]


def update_frontmatter(
    markdown_file: Path,
    author: str = "Matt Franchi",
    pub_datetime: str = "2025-01-15T12:00:00.000Z",
    mod_datetime: str = "",
    title: str = "Sensing AI Incidents & Risks from Global News",
    slug: str = "uair-brief",
    featured: bool = False,
    draft: bool = False,
    tags: List[str] | None = None,
    description: str = "An LLM-based Urban AI Risks (UAIR) assessment pipeline for extracting, verifying, and classifying risk information about AI use cases from large-scale news article collections. The pipeline processes articles through a five-stage workflow combining large language model inference, semantic verification, and regulatory classification.",
    bibliography: str = "sample.bib",
) -> None:
    """
    Replace the frontmatter in the generated markdown file with the specified frontmatter.
    """
    if tags is None:
        tags = [
            "artificial intelligence",
            "machine learning",
            "research",
            "risk assessment",
            "news analysis",
            "urban AI",
        ]
    
    # Read the markdown file
    content = markdown_file.read_text(encoding="utf-8")
    
    # Find and remove existing frontmatter (between --- markers)
    frontmatter_pattern = r"^---\n.*?\n---\n\n?"
    content = re.sub(frontmatter_pattern, "", content, flags=re.MULTILINE | re.DOTALL)
    
    # Build new frontmatter
    frontmatter_lines = [
        "---",
        f"author: {author}",
        f"pubDatetime: {pub_datetime}",
    ]
    
    if mod_datetime:
        frontmatter_lines.append(f"modDatetime: {mod_datetime}")
    else:
        frontmatter_lines.append("modDatetime:")
    
    frontmatter_lines.extend([
        f"title: {title}",
        f"slug: {slug}",
        f"featured: {str(featured).lower()}",
        f"draft: {str(draft).lower()}",
        "tags:",
    ])
    
    for tag in tags:
        frontmatter_lines.append(f"  - {tag}")
    
    frontmatter_lines.extend([
        f"description: {description}",
        f"bibliography: {bibliography}",
        "---",
        "",
    ])
    
    new_frontmatter = "\n".join(frontmatter_lines)
    
    # Write back with new frontmatter
    markdown_file.write_text(new_frontmatter + content, encoding="utf-8")


def update_asset_paths(markdown_file: Path, asset_base: str = "uair-brief") -> None:
    """
    Convert HTML img/embed tags to Markdown image syntax and update paths.
    Also converts PDF references to PNG versions.
    """
    content = markdown_file.read_text(encoding="utf-8")
    
    # First, update paths in HTML tags to use @assets/uair-brief/ format
    # Update img src paths: Images/filename.png -> @assets/uair-brief/filename.png
    content = re.sub(
        r'src="Images/([^"]+)"',
        rf'src="@assets/{asset_base}/\1"',
        content,
    )
    
    # Update embed src paths: figures/filename.pdf -> @assets/uair-brief/filename.png
    content = re.sub(
        r'src="figures/([^"]+)"',
        rf'src="@assets/{asset_base}/\1"',
        content,
    )
    
    # Replace .pdf with .png in paths
    content = re.sub(
        r'(@assets/[^"]+)\.pdf',
        r'\1.png',
        content,
    )
    
    # Now convert HTML tags to markdown
    # First, handle figures with captions - extract caption and convert embed to markdown
    def replace_figure_with_caption(match):
        embed_tag = match.group(1)
        caption_text = match.group(2).strip()
        
        # Extract src from embed tag
        src_match = re.search(r'src="([^"]+)"', embed_tag)
        if not src_match:
            return match.group(0)  # Return original if no src found
        
        src_path = src_match.group(1)
        
        # Use caption as alt text and title
        alt_text = caption_text.replace('\n', ' ').strip()
        return f'![{alt_text}]({src_path} "{alt_text}")'
    
    # Replace <figure>...</figure> blocks with markdown images
    content = re.sub(
        r'<figure[^>]*>\s*<embed\s+([^>]+)\s*/>\s*<figcaption>([^<]+)</figcaption>\s*</figure>',
        replace_figure_with_caption,
        content,
        flags=re.DOTALL,
    )
    
    # Convert <embed src="..." /> to markdown image (for embeds without figure wrapper)
    def replace_embed(match):
        src_path = match.group(1)
        # Extract filename for alt text
        filename = Path(src_path).stem.replace('_', ' ').replace('-', ' ').title()
        return f'![{filename}]({src_path} "{filename}")'
    
    content = re.sub(
        r'<embed\s+src="([^"]+)"[^>]*\s*/>',
        replace_embed,
        content,
    )
    
    # Convert <img src="..." alt="..." /> to markdown image
    def replace_img(match):
        src_path = match.group(1)
        alt_text = match.group(2) if match.group(2) else ""
        # If alt is generic like "image", use filename
        if alt_text.lower() in ("image", "img", ""):
            filename = Path(src_path).stem.replace('_', ' ').replace('-', ' ').title()
            alt_text = filename
        
        # Extract title from style if present, otherwise use alt
        title = alt_text
        return f'![{alt_text}]({src_path} "{title}")'
    
    content = re.sub(
        r'<img\s+src="([^"]+)"[^>]*alt="([^"]*)"[^>]*\s*/?>',
        replace_img,
        content,
    )
    
    # Also handle img tags without alt attribute
    content = re.sub(
        r'<img\s+src="([^"]+)"[^>]*\s*/?>',
        lambda m: f'![{Path(m.group(1)).stem.replace("_", " ").replace("-", " ").title()}]({m.group(1)} "{Path(m.group(1)).stem.replace("_", " ").replace("-", " ").title()}")',
        content,
    )
    
    markdown_file.write_text(content, encoding="utf-8")


def copy_assets(
    manuscript_dir: Path,
    output_dir: Path,
    asset_dirs: Iterable[str],
    public_dir: Path | None = None,
) -> None:
    """
    Copy supporting asset directories into a single assets/uair-brief folder.
    
    If public_dir is provided, copy all assets into a single directory structure
    suitable for copying into a website's /public/assets/uair-brief directory.
    """
    if public_dir:
        public_dir.mkdir(parents=True, exist_ok=True)
    
    for dirname in asset_dirs:
        src = manuscript_dir / dirname
        if not src.exists():
            continue

        # Copy next to markdown file (keep original structure for reference)
        dest = output_dir / dirname
        try:
            if dest.resolve() != src.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest, dirs_exist_ok=True)
        except FileNotFoundError:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)

        # Copy all files to unified assets directory if specified
        if public_dir:
            for file_path in src.rglob("*"):
                if file_path.is_file():
                    # Preserve relative path structure but flatten into single directory
                    rel_path = file_path.relative_to(src)
                    public_dest = public_dir / rel_path.name
                    # Handle name conflicts by keeping directory prefix
                    if public_dest.exists() and public_dest != file_path:
                        # Use subdirectory name as prefix if there's a conflict
                        public_dest = public_dir / f"{dirname}_{rel_path.name}"
                    public_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, public_dest)
            print(f"Copied {dirname}/ files to {public_dir}")


def build_pandoc_command(
    pandoc_cmd: List[str],
    main_file: Path,
    output_file: Path,
    output_format: str,
    manuscript_dir: Path,
    resource_dirs: Iterable[str],
    bibliography: Path | None,
    metadata_title: str | None,
) -> List[str]:
    """
    Build the pandoc command for conversion.
    """
    cmd = [
        *pandoc_cmd,
        str(main_file.name),
        "--from=latex",
        f"--to={output_format}",
        "--standalone",
        "--number-sections",
        "--citeproc",
        "--wrap=auto",
        "--top-level-division=section",
        "--output",
        str(output_file),
    ]

    if metadata_title:
        cmd.extend(["--metadata", f"title={metadata_title}"])

    if bibliography and bibliography.exists():
        cmd.extend(["--bibliography", bibliography.name])

    resource_entries = [manuscript_dir]
    for dirname in resource_dirs:
        path = manuscript_dir / dirname
        if path.exists():
            resource_entries.append(path)

    if resource_entries:
        resource_path = ":".join(str(path) for path in resource_entries)
        cmd.append(f"--resource-path={resource_path}")

    return cmd


def convert(
    main_file: Path,
    output_file: Path,
    output_format: str,
    manuscript_dir: Path,
    resource_dirs: Iterable[str],
    bibliography: Path | None,
    metadata_title: str | None,
) -> None:
    """
    Execute pandoc conversion with LaTeX input expansion.
    """
    pandoc_cmd = ensure_pandoc()
    
    # Expand all \input commands in the main file
    expanded_content = expand_latex_inputs(main_file, manuscript_dir)
    
    # Create a temporary file with expanded content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, dir=manuscript_dir, encoding='utf-8') as tmp_file:
        tmp_file.write(expanded_content)
        tmp_file_path = Path(tmp_file.name)
    
    try:
        cmd = build_pandoc_command(
            pandoc_cmd=pandoc_cmd,
            main_file=tmp_file_path,
            output_file=output_file,
            output_format=output_format,
            manuscript_dir=manuscript_dir,
            resource_dirs=resource_dirs,
            bibliography=bibliography,
            metadata_title=metadata_title,
        )
        
        # Replace the temp file name in the command
        cmd[cmd.index(str(tmp_file_path.name))] = str(tmp_file_path.name)
        
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=manuscript_dir, check=True)
    finally:
        # Clean up temporary file
        tmp_file_path.unlink()


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the UAIR brief LaTeX manuscript into Markdown using pandoc."
    )
    parser.add_argument(
        "--manuscript-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_DIR,
        help="Directory containing MAIN.tex and supporting assets.",
    )
    parser.add_argument(
        "--main-file",
        type=Path,
        default=DEFAULT_MAIN_FILE,
        help="Path to the root LaTeX file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination Markdown file.",
    )
    parser.add_argument(
        "--format",
        default="gfm",
        help="Pandoc output format (default: gfm).",
    )
    parser.add_argument(
        "--resource-dirs",
        nargs="*",
        default=list(DEFAULT_RESOURCE_DIRS),
        help="Additional manuscript subdirectories to expose via pandoc's --resource-path.",
    )
    parser.add_argument(
        "--bibliography",
        type=Path,
        default=DEFAULT_MANUSCRIPT_DIR / "sample.bib",
        help="Bibliography file to include if present.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Override the document title metadata.",
    )
    parser.add_argument(
        "--copy-assets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy asset directories (images, figures, tables) next to the Markdown output.",
    )
    parser.add_argument(
        "--asset-dirs",
        nargs="*",
        default=list(DEFAULT_ASSET_DIRS),
        help="Manuscript subdirectories to copy when --copy-assets is enabled.",
    )
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=DEFAULT_PUBLIC_DIR,
        help="Directory to copy assets for website public directory (e.g., assets/uair-brief/). "
        "Defaults to papers/uair_brief/public/assets/uair-brief/. "
        "Set to empty string to disable.",
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    manuscript_dir = args.manuscript_dir.expanduser().resolve()
    main_file = (
        args.main_file
        if args.main_file.is_absolute()
        else manuscript_dir / args.main_file
    )
    output_file = args.output.expanduser().resolve()

    if not main_file.exists():
        print(f"Main LaTeX file not found: {main_file}", file=sys.stderr)
        return 1

    output_file.parent.mkdir(parents=True, exist_ok=True)

    bibliography = (
        args.bibliography
        if args.bibliography.is_absolute()
        else manuscript_dir / args.bibliography
    )

    try:
        convert(
            main_file=main_file,
            output_file=output_file,
            output_format=args.format,
            manuscript_dir=manuscript_dir,
            resource_dirs=args.resource_dirs,
            bibliography=bibliography,
            metadata_title=args.title,
        )
    except PandocUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Pandoc failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    # Update frontmatter with website-specific metadata
    update_frontmatter(
        markdown_file=output_file,
        title=args.title or "Sensing AI Incidents & Risks from Global News",
        bibliography=bibliography.name if bibliography.exists() else "sample.bib",
    )
    
    # Update asset paths to use @assets/uair-brief/ format
    update_asset_paths(output_file)
    
    # Add appendices content if it's missing
    # Pandoc may not process custom environments, so we add it manually
    content = output_file.read_text(encoding="utf-8")
    if "# Appendices" in content and "<div class=\"appendices\">" in content:
        # Check if appendices section is empty
        appendices_match = re.search(r'# Appendices\s+<div class="appendices">\s*</div>', content, re.DOTALL)
        if appendices_match:
            # Expand appendices content
            supplement_file = manuscript_dir / "sections" / "A_supplement.tex"
            if supplement_file.exists():
                appendices_latex = expand_latex_inputs(supplement_file, manuscript_dir)
                # Remove LaTeX formatting commands that pandoc can't handle
                # Remove \titleformat commands line by line (simpler and safer)
                lines = appendices_latex.split('\n')
                filtered_lines = []
                i = 0
                while i < len(lines):
                    line = lines[i]
                    stripped = line.strip()
                    # Skip \titleformat lines (they span multiple lines with braces)
                    if stripped.startswith('\\titleformat'):
                        # Skip this line and count braces to find the end
                        brace_count = line.count('{') - line.count('}')
                        i += 1
                        # Continue skipping until braces are balanced
                        while i < len(lines) and brace_count > 0:
                            brace_count += lines[i].count('{') - lines[i].count('}')
                            i += 1
                        continue
                    filtered_lines.append(line)
                    i += 1
                appendices_latex = '\n'.join(filtered_lines)
                # Wrap in LaTeX document
                wrapped_latex = (
                    "\\documentclass{article}\n"
                    "\\begin{document}\n"
                    f"{appendices_latex}\n"
                    "\\end{document}"
                )
                # Convert appendices LaTeX to markdown using pandoc
                with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, dir=manuscript_dir, encoding='utf-8') as tmp_tex:
                    tmp_tex.write(wrapped_latex)
                    tmp_tex_path = Path(tmp_tex.name)
                
                try:
                    pandoc_cmd = ensure_pandoc()
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, dir=manuscript_dir, encoding='utf-8') as tmp_md:
                        tmp_md_path = Path(tmp_md.name)
                    
                    try:
                        result = subprocess.run(
                            pandoc_cmd + [
                                str(tmp_tex_path.name),
                                "--from=latex",
                                "--to=gfm",
                                "--wrap=auto",
                                "--output", str(tmp_md_path),
                            ],
                            cwd=manuscript_dir,
                            check=False,
                            capture_output=True,
                        )
                        if result.returncode == 0 and tmp_md_path.exists():
                            appendices_md = tmp_md_path.read_text(encoding="utf-8")
                            # Remove document wrapper if present
                            appendices_md = re.sub(r'^.*?\\begin\{document\}', '', appendices_md, flags=re.DOTALL)
                            appendices_md = re.sub(r'\\end\{document\}.*$', '', appendices_md, flags=re.DOTALL)
                            # Convert to markdown image syntax (will be processed by update_asset_paths later)
                            appendices_md = re.sub(r'src="Images/([^"]+)"', r'src="@assets/uair-brief/\1"', appendices_md)
                            appendices_md = re.sub(r'src="figures/([^"]+)"', r'src="@assets/uair-brief/\1"', appendices_md)
                            # Replace .pdf with .png
                            appendices_md = appendices_md.replace('.pdf', '.png')
                            # Replace empty appendices section with content
                            # Use a lambda to avoid regex interpretation of replacement string
                            def replace_appendices(match):
                                return f'# Appendices\n\n<div class="appendices">\n\n{appendices_md}\n\n</div>'
                            content = re.sub(
                                r'# Appendices\s+<div class="appendices">\s*</div>',
                                replace_appendices,
                                content,
                                flags=re.DOTALL,
                            )
                            output_file.write_text(content, encoding="utf-8")
                            # Update asset paths again after adding appendices
                            update_asset_paths(output_file)
                        else:
                            error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "No error message"
                            print(f"Warning: Could not convert appendices LaTeX to markdown (exit code {result.returncode}).", file=sys.stderr)
                            print(f"Pandoc error: {error_msg[:300]}", file=sys.stderr)
                    finally:
                        if tmp_md_path.exists():
                            tmp_md_path.unlink()
                finally:
                    tmp_tex_path.unlink()
            else:
                print(f"Warning: Supplement file not found: {supplement_file}", file=sys.stderr)
        else:
            print("Warning: Appendices section pattern did not match", file=sys.stderr)

    if args.copy_assets:
        # Handle empty string as None (to disable public dir)
        public_dir = None
        if args.public_dir and str(args.public_dir):
            public_dir = args.public_dir.expanduser().resolve()
        
        copy_assets(
            manuscript_dir=manuscript_dir,
            output_dir=output_file.parent,
            asset_dirs=args.asset_dirs,
            public_dir=public_dir,
        )

    print(f"Wrote Markdown to {output_file}")
    if args.public_dir and str(args.public_dir):
        public_path = args.public_dir.expanduser().resolve()
        print(f"Assets copied to {public_path}")
        print(f"Copy the contents of this directory to your website's /public/assets/uair-brief/ directory")
    return 0


if __name__ == "__main__":
    sys.exit(main())

