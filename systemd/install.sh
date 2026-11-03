#!/bin/sh
# Install the speaker as systemd user services (no Nix needed).
#   systemd/install.sh            install or update the units
#   systemd/install.sh uninstall  stop and remove them (keeps ~/.config/speaker/env)
set -eu

DIR=$(cd "$(dirname "$0")/.." && pwd)
UNITS="speaker.target speaker-llm.service speaker-embed.service speaker-asr.service speaker-brain.service speaker-voice.service"
CONFIG=${XDG_CONFIG_HOME:-$HOME/.config}
UNIT_DIR=$CONFIG/systemd/user
ENV_FILE=$CONFIG/speaker/env

if [ "${1:-}" = uninstall ]; then
    systemctl --user disable --now speaker.target 2>/dev/null || true
    for u in $UNITS; do rm -f "$UNIT_DIR/$u"; done
    systemctl --user daemon-reload
    echo "Removed the units; $ENV_FILE is kept."
    exit 0
fi

[ -x "$DIR/.venv/bin/speaker" ] || { echo "Run 'uv sync' in $DIR first." >&2; exit 1; }

mkdir -p "$UNIT_DIR" "$(dirname "$ENV_FILE")"
for u in $UNITS; do
    sed "s|@DIR@|$DIR|g" "$DIR/systemd/$u" > "$UNIT_DIR/$u"
done
if [ ! -e "$ENV_FILE" ]; then
    sed "s|@DIR@|$DIR|g" "$DIR/systemd/speaker.env.example" > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "Wrote $ENV_FILE: check PATH, LOCALVQE_LIB and ALSA_CONFIG_PATH there."
fi
systemctl --user daemon-reload
echo "Installed. Start now and at login:  systemctl --user enable --now speaker.target"
echo "Start at boot without logging in:   sudo loginctl enable-linger $USER"
echo "Logs:                               journalctl --user -u speaker-voice -f"
