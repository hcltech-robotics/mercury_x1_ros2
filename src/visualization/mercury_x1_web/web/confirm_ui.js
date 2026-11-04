// In-page confirmation dialog shared by the control page and the head display, instead of window.confirm():
// the kiosk browser (surf) and some tabs block or mis-render native popups, and they cannot be styled or touched.
//   const yes = await ask("Power the arms off?", { ok: "Power off", danger: true });
(function () {
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  let root = null, pending = null;

  function build() {
    const style = document.createElement("style");
    style.textContent = `
      #ask-overlay { position: fixed; inset: 0; z-index: 1000; display: none; align-items: center; justify-content: center;
        background: rgba(0, 0, 0, .55); padding: 16px; }
      #ask-overlay.show { display: flex; }
      #ask-box { background: var(--panel, #1a1d23); color: var(--text, #eef0f3); border: 1px solid var(--line, #323743);
        border-radius: 14px; padding: 20px 22px; width: min(520px, 100%); box-shadow: 0 12px 40px rgba(0, 0, 0, .5); }
      #ask-msg { white-space: pre-wrap; line-height: 1.4; font-size: 1.05em; margin-bottom: 18px; }
      #ask-btns { display: flex; gap: 12px; justify-content: flex-end; }
      #ask-btns button { font: inherit; color: var(--text, #eef0f3); background: var(--panel2, #242830);
        border: 1px solid var(--line, #323743); border-radius: 12px; padding: 12px 20px; min-width: 110px; cursor: pointer; }
      #ask-btns button.ok { background: #134a66; border-color: var(--accent, #4cc2ff); }
      #ask-btns button.ok.danger { background: #5a1f26; border-color: var(--bad, #ff5468); }`;
    document.head.appendChild(style);
    root = document.createElement("div");
    root.id = "ask-overlay";
    root.innerHTML = `<div id="ask-box" role="dialog" aria-modal="true"><div id="ask-msg"></div>
      <div id="ask-btns"><button class="cancel"></button><button class="ok"></button></div></div>`;
    document.body.appendChild(root);
    root.querySelector(".cancel").onclick = () => finish(false);
    root.querySelector(".ok").onclick = () => finish(true);
    root.onclick = (e) => { if (e.target === root) finish(false); };
    document.addEventListener("keydown", (e) => {
      if (!pending) return;
      if (e.key === "Escape") finish(false);
      if (e.key === "Enter") finish(true);
    });
  }

  function finish(result) {
    if (!pending) return;
    const p = pending; pending = null;
    root.classList.remove("show");
    p(result);
  }

  // -> Promise<boolean>; opts: ok / cancel button labels, danger (red OK button)
  window.ask = function (message, opts = {}) {
    if (!root) build();
    if (pending) finish(false);
    root.querySelector("#ask-msg").innerHTML = esc(message);
    const ok = root.querySelector(".ok"), cancel = root.querySelector(".cancel");
    ok.textContent = opts.ok || "OK"; cancel.textContent = opts.cancel || "Cancel";
    ok.classList.toggle("danger", !!opts.danger);
    root.classList.add("show");
    return new Promise((resolve) => { pending = resolve; setTimeout(() => cancel.focus(), 0); });
  };
})();
