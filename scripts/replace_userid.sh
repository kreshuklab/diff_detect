#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 OLD_USERNAME NEW_USERNAME" >&2
    exit 2
fi

OLD_USERNAME=$1
NEW_USERNAME=$2

cp database.db database.db.bak

sqlite3 database.db "
BEGIN;
UPDATE user SET id='$NEW_USERNAME', name='$NEW_USERNAME' WHERE id='$OLD_USERNAME';
UPDATE explainoutcome SET user='$NEW_USERNAME' WHERE user='$OLD_USERNAME';
UPDATE rateoutcome SET own='$NEW_USERNAME' WHERE own='$OLD_USERNAME';
UPDATE rateoutcome SET peer='$NEW_USERNAME' WHERE peer='$OLD_USERNAME';
UPDATE rateoutcome SET ai='$NEW_USERNAME' WHERE ai='$OLD_USERNAME';
UPDATE rateoutcome SET most_convincing='$NEW_USERNAME' WHERE most_convincing='$OLD_USERNAME';
UPDATE rateoutcome SET most_likely_ai='$NEW_USERNAME' WHERE most_likely_ai='$OLD_USERNAME';
COMMIT;
"

sqlite3 database.db "SELECT id, name, lab, kind, role FROM user ORDER BY id;"
