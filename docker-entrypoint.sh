#!/bin/sh
set -eu

# Keep runtime-created SQLite files writable when OpenShift rotates the
# arbitrary UID while retaining the image's root-group permission model.
umask 0002

# Mirror the server's default only for permission preparation. Keep DATABASE_URL
# unset so the Rust CLI remains authoritative for the actual connection URL.
agentic_api_home=${AGENTIC_API_HOME:-/var/lib/agentic-api}
database_url=${DATABASE_URL:-sqlite://${agentic_api_home}/agentic_api.db}
case "$database_url" in
    sqlite::memory:*)
        ;;
    sqlite://*)
        case "$database_url" in
            *%*)
                echo "DATABASE_URL percent-encoded SQLite paths are not supported by this container entrypoint" >&2
                exit 64
                ;;
        esac
        sqlite_query=
        case "$database_url" in
            *\?*)
                sqlite_query=${database_url#*\?}
                sqlite_query=${sqlite_query%%\#*}
                ;;
        esac
        prepare_sqlite=true
        case "&$sqlite_query&" in
            *"&mode=ro&"* | *"&mode=rw&"* | *"&mode=memory&"*)
                prepare_sqlite=false
                ;;
        esac

        if [ "$prepare_sqlite" = true ]; then
            # `?` and `#` are URI query/fragment delimiters, so they are not part
            # of the filesystem path extracted here. Keep the parent directory
            # group-writable so a rotated arbitrary UID can create SQLite sidecar files.
            database_path=${database_url#sqlite://}
            database_path=${database_path%%\?*}
            database_path=${database_path%%\#*}
            if [ -n "$database_path" ]; then
                if [ ! -e "$database_path" ]; then
                    : >"$database_path"
                    chmod g+rw "$database_path"
                fi
                database_directory=$(dirname -- "$database_path")
                chmod g+rwx "$database_directory" 2>/dev/null || true
            fi
        fi
        ;;
esac

# Both binaries share the storage prepared above. The command may name the one to
# run; anything else is passed to the default. AGENTIC_BINARY is the alternative
# for platforms that set arguments but not the command.
case "${1:-}" in
    agentic-server | agentic-llm-d)
        binary=$1
        shift
        ;;
    *)
        binary=${AGENTIC_BINARY:-agentic-server}
        ;;
esac

exec "$binary" "$@"
