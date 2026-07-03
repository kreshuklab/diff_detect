import json

import streamlit as st


def patch_canvas_toolbar(run_key: str | None = None) -> None:
    """Adjust drawable-canvas toolbar icon colors for the active Streamlit theme."""
    run_key_json = json.dumps(run_key or "")
    script = """
                <script>
                (function() {
                    const PATCH_RUN_ID = __PATCH_RUN_ID__;
                    const STYLE_ID = 'diff-detect-toolbar-theme-style';
                    const OBSERVER_KEY = '__diffDetectCanvasToolbarObserver';
                    const PATCH_FN_KEY = '__diffDetectPatchCanvasToolbar';
                    const RETRY_DELAYS = [0, 50, 150, 300, 600, 1000, 1500, 2500];
                    let applyScheduled = false;

                    function parseRgb(value) {
                        const match = String(value || '').match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/i);
                        if (!match) return null;
                        return [Number(match[1]), Number(match[2]), Number(match[3])];
                    }

                    function isDarkTheme() {
                        try {
                            const parentDoc = window.parent.document;
                            const app = parentDoc.querySelector('.stApp') || parentDoc.body;
                            const bg = getComputedStyle(app).backgroundColor || getComputedStyle(parentDoc.body).backgroundColor;
                            const rgb = parseRgb(bg);
                            if (!rgb) return false;
                            const luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
                            return luminance < 140;
                        } catch (_) {
                            return false;
                        }
                    }

                    function apply() {
                        const applyDarkFix = isDarkTheme();
                        const frames = window.parent.document.querySelectorAll(
                            'iframe[title="streamlit_drawable_canvas.st_canvas"]'
                        );

                        frames.forEach((frame) => {
                            try {
                                if (!frame.dataset.diffDetectToolbarLoadWatcher) {
                                    frame.addEventListener('load', scheduleApply);
                                    frame.dataset.diffDetectToolbarLoadWatcher = 'true';
                                }

                                const doc = frame.contentDocument || frame.contentWindow?.document;
                                if (!doc || !doc.head) return;

                                let style = doc.getElementById(STYLE_ID);
                                if (!style) {
                                    style = doc.createElement('style');
                                    style.id = STYLE_ID;
                                    doc.head.appendChild(style);
                                }

                                let cssText = `
                                    img[alt="Send to Streamlit"] {
                                        display: none !important;
                                    }
                                `;
                                if (applyDarkFix) {
                                    cssText += `
                                        img[alt="Undo"],
                                        img[alt="Redo"],
                                        img[alt="Reset canvas & history"] {
                                            filter: invert(1);
                                        }
                                    `;
                                }
                                if (style.textContent !== cssText) {
                                    style.textContent = cssText;
                                }
                                frame.dataset.diffDetectToolbarPatchRun = PATCH_RUN_ID;
                            } catch (_) {
                                // Ignore cross-frame timing/access issues.
                            }
                        });
                    }

                    function scheduleApply() {
                        if (applyScheduled) return;
                        applyScheduled = true;
                        RETRY_DELAYS.forEach((delay, index) => {
                            setTimeout(() => {
                                apply();
                                if (index === RETRY_DELAYS.length - 1) {
                                    applyScheduled = false;
                                }
                            }, delay);
                        });
                    }

                    function installWatcher() {
                        try {
                            window.parent[PATCH_FN_KEY] = scheduleApply;
                            if (window.parent[OBSERVER_KEY]) return;

                            const root = window.parent.document.body;

                            const observer = new MutationObserver(() => {
                                const patch = window.parent[PATCH_FN_KEY];
                                if (typeof patch === 'function') {
                                    patch();
                                }
                            });

                            observer.observe(root, {
                                childList: true,
                                subtree: true,
                                attributes: true,
                                attributeFilter: ['class', 'style', 'title', 'src'],
                            });
                            window.parent[OBSERVER_KEY] = observer;
                        } catch (_) {
                            // Ignore watcher setup failures.
                        }
                    }

                    installWatcher();
                    scheduleApply();
                })();
                </script>
                """.replace("__PATCH_RUN_ID__", run_key_json)
    st.html(
        script,
        width="content",
        unsafe_allow_javascript=True,
    )
