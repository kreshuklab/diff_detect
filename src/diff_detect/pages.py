from ._builder import PageBuilder


def build_pages():
    page_builder = PageBuilder()
    return list(page_builder.pages.values())
