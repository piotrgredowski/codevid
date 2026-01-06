"""Mouse helper script for visible cursor in Playwright recordings.

Playwright's video recording uses CDP which does NOT capture the system cursor.
This module provides a JavaScript injection that creates a visible cursor element
that follows mouse events and will be captured in the video.

The cursor is wrapped in a Shadow DOM with a custom element tag to avoid
affecting XPath selectors that use positional indices (e.g., //div[6]).
"""

MOUSE_HELPER_JS = """
(function() {
    // Only install once and only in top frame
    if (window.__codevidCursorInstalled || window !== window.parent) return;
    window.__codevidCursorInstalled = true;

    function installCursor() {
        // Check if already installed
        if (document.querySelector('codevid-cursor-host')) return;

        // Create a custom element host (not a div, won't affect //div[N] XPath)
        const host = document.createElement('codevid-cursor-host');
        host.style.cssText = 'position:fixed;top:0;left:0;width:0;height:0;pointer-events:none;z-index:2147483647;';

        // Attach Shadow DOM to isolate cursor from document queries
        const shadow = host.attachShadow({ mode: 'closed' });

        // Create cursor inside Shadow DOM
        const cursor = document.createElement('div');
        cursor.style.cssText = `
            pointer-events: none;
            position: fixed;
            z-index: 2147483647;
            width: 24px;
            height: 24px;
            background: rgba(255, 220, 50, 0.85);
            border: 2px solid rgba(0, 0, 0, 0.6);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: transform 0.08s ease-out, background 0.15s;
            box-shadow: 0 0 8px rgba(255, 220, 50, 0.5);
            left: -100px;
            top: -100px;
        `;
        shadow.appendChild(cursor);
        document.body.appendChild(host);

        document.addEventListener('mousemove', e => {
            cursor.style.left = e.clientX + 'px';
            cursor.style.top = e.clientY + 'px';
        }, true);

        document.addEventListener('mousedown', e => {
            cursor.style.transform = 'translate(-50%, -50%) scale(0.7)';
            cursor.style.background = 'rgba(255, 180, 0, 0.95)';
        }, true);

        document.addEventListener('mouseup', e => {
            cursor.style.transform = 'translate(-50%, -50%) scale(1)';
            cursor.style.background = 'rgba(255, 220, 50, 0.85)';
        }, true);
    }

    // Install immediately if DOM ready, otherwise wait
    if (document.body) {
        installCursor();
    } else {
        document.addEventListener('DOMContentLoaded', installCursor);
    }
})();
"""
