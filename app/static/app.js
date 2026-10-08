/* Kiosk + enrollment polling. No framework, no build step - this runs on the
   Pi itself and the whole point is that it stays cheap.

   Note what the feed does NOT do: it never renders a face image at all. The
   payload it reads comes from /api/live, which the kiosk polls without a
   login, so it carries names and initials only. An initials avatar is enough
   to confirm the right person was recorded without putting biometric images
   on a screen the whole lobby can read. */

(function () {
  "use strict";

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday"];

  function el(id) { return document.getElementById(id); }
  function pad(n) { return n < 10 ? "0" + n : "" + n; }

  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;",
               '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function setText(id, value) {
    var node = el(id);
    if (node) node.textContent = value;
  }

  function tick() {
    var clock = el("clock");
    if (!clock) return;
    var now = new Date();
    clock.textContent = pad(now.getHours()) + ":" + pad(now.getMinutes());
    var date = el("date");
    if (date) {
      date.textContent = DAYS[now.getDay()] + ", " + now.getDate() + " " +
        MONTHS[now.getMonth()] + " " + now.getFullYear();
    }
  }

  /* ------------------------------------------------------------- kiosk feed */

  var lastEventId = null;
  var greetTimer = null;

  function greet(name, when, kind) {
    var box = el("greeting");
    if (!box) return;
    var leaving = kind === "check_out";
    el("greet-name").textContent = (leaving ? "Goodbye, " : "Welcome, ") + name;
    el("greet-sub").textContent =
      (leaving ? "Checked out at " : "Checked in at ") + when;
    box.classList.toggle("leaving", leaving);
    box.classList.add("show");
    clearTimeout(greetTimer);
    greetTimer = setTimeout(function () { box.classList.remove("show"); }, 4000);
  }

  function renderFeed(events) {
    var feed = el("feed");
    if (!feed) return;
    if (!events.length) {
      feed.innerHTML = '<div class="empty">No activity yet today</div>';
      return;
    }
    feed.innerHTML = events.map(function (e) {
      var time = (e.ts || "").slice(11, 16);
      var inbound = e.kind !== "check_out";
      // Always an initials avatar. /api/live is polled by the kiosk, which
      // needs no login, so this payload must never carry a face image.
      var face = '<span class="av c' + (e.avatar == null ? "" : e.avatar) +
        (e.person_id ? '' : ' unknown') + '">' +
        escapeHtml(e.initials || "?") + '</span>';
      var meta = e.department || e.emp_code || "";
      return '<div class="ev">' + face +
             '<div class="grow"><div class="nm">' + escapeHtml(e.name) + '</div>' +
             (meta ? '<div class="mt">' + escapeHtml(meta) + '</div>' : '') +
             '</div>' +
             '<span class="pill ' + (inbound ? 'check_in' : 'check_out') + '">' +
             (inbound ? 'In' : 'Out') + '</span>' +
             '<span class="tm">' + time + '</span></div>';
    }).join("");
  }

  function pollLive() {
    fetch("/api/live", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        setText("c-inside", d.summary.inside);
        setText("c-present", d.summary.present);
        setText("c-absent", d.summary.absent);
        setText("c-late", d.summary.late);

        var badge = el("status-badge");
        if (badge) {
          badge.className = "tag " + (d.recognizer_up ? "" : "off");
          badge.innerHTML = '<span class="dotmark"></span>' +
            (d.recognizer_up ? "Live" : "Camera offline");
        }

        renderFeed(d.events || []);

        if (d.events && d.events.length) {
          var newest = d.events[0];
          if (lastEventId !== null && newest.id !== lastEventId &&
              newest.person_id) {
            greet(newest.name, (newest.ts || "").slice(11, 16), newest.kind);
          }
          lastEventId = newest.id;
        }
      })
      .catch(function () { /* recognizer or network hiccup - try again */ });
  }

  /* -------------------------------------------------------- enroll progress */

  function pollEnroll(cmdId) {
    fetch("/api/enroll/" + cmdId, { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var bar = el("bar");
        if (bar) bar.style.width = (d.progress || 0) + "%";
        setText("hint", d.message || "Waiting for the camera...");
        setText("pct", (d.progress || 0) + "%");

        if (d.status === "done") {
          setText("hint", d.message || "Done");
          var ok = el("done-actions");
          if (ok) ok.style.display = "flex";
          return;
        }
        if (d.status === "error") {
          var box = el("err");
          if (box) {
            box.style.display = "flex";
            box.textContent = d.message || "Enrollment failed";
          }
          var retry = el("retry-actions");
          if (retry) retry.style.display = "flex";
          return;
        }
        setTimeout(function () { pollEnroll(cmdId); }, 700);
      })
      .catch(function () {
        setTimeout(function () { pollEnroll(cmdId); }, 1500);
      });
  }

  /* ------------------------------------------------------------------- boot */

  document.addEventListener("DOMContentLoaded", function () {
    if (el("clock")) { tick(); setInterval(tick, 1000); }
    if (el("feed")) { pollLive(); setInterval(pollLive, 2000); }

    var cap = document.body.getAttribute("data-enroll");
    if (cap) pollEnroll(cap);

    // Confirm-before-destroy on any form that asks for it.
    document.querySelectorAll("form[data-confirm]").forEach(function (f) {
      f.addEventListener("submit", function (ev) {
        if (!window.confirm(f.getAttribute("data-confirm"))) ev.preventDefault();
      });
    });
  });
})();
