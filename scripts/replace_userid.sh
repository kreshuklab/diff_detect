#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 2 ]; then
    OLD_USERNAME=$1
    NEW_USERNAME=$2
    OLD_USER_ID=$OLD_USERNAME
    NEW_USER_ID=$NEW_USERNAME
    NEW_LAB_SQL="NULL"
elif [ "$#" -eq 4 ]; then
    OLD_LAB=$1
    OLD_USERNAME=$2
    NEW_LAB=$3
    NEW_USERNAME=$4
    OLD_USER_ID="$OLD_LAB/$OLD_USERNAME"
    NEW_USER_ID="$NEW_LAB/$NEW_USERNAME"
    NEW_LAB_SQL="'$NEW_LAB'"
else
    echo "Usage: $0 OLD_USERNAME NEW_USERNAME" >&2
    echo "   or: $0 OLD_LAB OLD_USERNAME NEW_LAB NEW_USERNAME" >&2
    exit 2
fi

cp database.db database.db.bak

sqlite3 database.db "
BEGIN;
UPDATE user SET id='$NEW_USER_ID', lab=$NEW_LAB_SQL, name='$NEW_USERNAME' WHERE id='$OLD_USER_ID';
UPDATE explainoutcome SET user='$NEW_USER_ID' WHERE user='$OLD_USER_ID';
UPDATE rateoutcome SET own='$NEW_USER_ID' WHERE own='$OLD_USER_ID';
UPDATE rateoutcome SET peer='$NEW_USER_ID' WHERE peer='$OLD_USER_ID';
UPDATE rateoutcome SET ai='$NEW_USER_ID' WHERE ai='$OLD_USER_ID';
UPDATE rateoutcome SET most_convincing='$NEW_USER_ID' WHERE most_convincing='$OLD_USER_ID';
UPDATE rateoutcome SET most_likely_ai='$NEW_USER_ID' WHERE most_likely_ai='$OLD_USER_ID';
COMMIT;
"

sqlite3 database.db "SELECT id, name, lab, kind, role FROM user ORDER BY id;"
