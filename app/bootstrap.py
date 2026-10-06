"""One-time setup: create the schema, create the first admin, sanity-check models.

Called by run.sh. Safe to run repeatedly - it never overwrites an existing
admin or an existing database.
"""
import os
import sys

from . import auth, config, db


def check_models():
    problems = []
    for key in ("detect.model", "embed.model"):
        path = config.abspath(config.g(key))
        if not os.path.exists(path):
            problems.append(path)
    return problems


def main():
    conn = db.init()
    print("[bootstrap] database ready at %s" % db.path())

    for folder in ("paths.unknowns", "paths.faces"):
        os.makedirs(config.abspath(config.g(folder)), exist_ok=True)
    os.makedirs(config.abspath("data/thumbs"), exist_ok=True)

    generated = auth.ensure_admin(conn)
    if generated:
        print("")
        print("  ============================================================")
        print("   FIRST ADMIN ACCOUNT CREATED")
        print("     username: admin")
        print("     password: %s" % generated)
        print("   This is shown ONCE. Log in and change it immediately.")
        print("  ============================================================")
        print("")
    else:
        print("[bootstrap] admin account already exists")

    missing = check_models()
    if missing:
        print("[bootstrap] WARNING - these models are missing:")
        for path in missing:
            print("    %s" % path)
        print("    Run: ./run.sh models")
        conn.close()
        return 1

    counts = conn.execute(
        "SELECT (SELECT COUNT(*) FROM people) AS p, "
        "(SELECT COUNT(*) FROM templates) AS t").fetchone()
    print("[bootstrap] %d people, %d face templates enrolled"
          % (counts["p"], counts["t"]))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
