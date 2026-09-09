"""
GuideML Command-Line Interface (CLI)
====================================
Launches GuideML locally with a single command and auto-opens the browser.
Usage:
    guideml [--port 5000] [--no-browser]
"""

import os
import sys
import argparse
import webbrowser
import threading
import time


def main():
    parser = argparse.ArgumentParser(
        description="GuideML — Elite Tabular AutoML Recommender System"
    )
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.environ.get('PORT', 5000)),
        help="Port to run the GuideML server on (default: 5000)"
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help="Host interface to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help="Do not automatically open the browser"
    )
    args = parser.parse_args()

    # Import the app
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from interface.app import app

    target_url = f"http://{args.host}:{args.port}"
    print("\n" + "=" * 60)
    print("  🚀 Starting GuideML AutoML Engine")
    print(f"  🌐 Server URL: {target_url}")
    print("=" * 60 + "\n")

    if not args.no_browser:
        def _open_tab():
            time.sleep(1.2)
            try:
                webbrowser.open(target_url)
            except Exception:
                pass
        threading.Thread(target=_open_tab, daemon=True).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
