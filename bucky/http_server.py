from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import threading
from typing import Any
from bucky.message_utils import resolve_content, ContentType
from langchain_core.messages import BaseMessage

HTML_PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Agent State</title>
</head>
<body style="font-family: 'Helvetica Neue', Helvetica, sans-serif; font-size: 16px; width:80%; margin-left:auto; margin-right:auto;">
{messages}
</body>
</html>
"""

HTML_MESSAGE_TEMPLATE = """<p>
<h3>{message_type} Message</h3>
{content}
</p>
<hr>"""


def to_html(data: Any) -> str:
    if isinstance(data, dict):
        rows = []
        for key, value in data.items():
            rows.append(f"<tr><td align='right'>{key.title()}:</td><td>{value}</td></tr>")
        return f"<table border='0'>{''.join(rows)}</table>"
    return f"{data}".strip().replace("\n", "<br>")


class AgentStateHttpServerHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, server_instance, **kwargs):
        self.server_instance: AgentStateHttpServer = server_instance
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        pass

    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            html_messages: list[str] = []
            for html_message in self.server_instance.get_html_messages():
                html_messages.append(html_message)

            if not html_messages:
                html_messages.append("state is empty")

            html_content = HTML_PAGE_TEMPLATE.format(messages="\n".join(html_messages))
            self.wfile.write(html_content.encode("utf-8"))
        else:
            self.send_error(404, "Endpoint Not Found")


class AgentStateHttpServer(HTTPServer):
    def __init__(self, host="", port=8000, handler_class=AgentStateHttpServerHandler):
        self._lock = threading.RLock()
        self._html_messages: list[str] = []
        super().__init__((host, port), lambda *args, **kwargs: handler_class(*args, server_instance=self, **kwargs))

    def get_html_messages(self) -> list[str]:
        with self._lock:
            return self._html_messages

    def set_agent_state(self, state: list[BaseMessage]):
        html_messages: list[str] = []
        for message in state:
            content: list[str] = []
            for content_item in resolve_content(message):
                if content_item.type == ContentType.IMAGE:
                    content.append(f'<p><img src="{str(content_item.data)}" /></p>')
                else:
                    content.append(f"<p>{to_html(content_item.data)}</p>")
            html_messages.append(
                HTML_MESSAGE_TEMPLATE.format(message_type=message.type.title(),
                                             content="\n".join(content)))
        with self._lock:
            self._html_messages = html_messages

    def start(self):
        """Start the server in a non-blocking thread"""
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()


if __name__ == "__main__":
    server = AgentStateHttpServer(port=8000)
    server.start()
    input("Press Enter to stop the server...\n")
