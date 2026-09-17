import reflex as rx

config = rx.Config(
    app_name="tax_qa_web",
    app_module_import="tax_qa.web",
    disable_plugins=[rx.plugins.SitemapPlugin],
)
