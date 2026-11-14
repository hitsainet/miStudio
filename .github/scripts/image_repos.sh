# The Docker Hub repository each image is pushed to (ENH-009 042 FR-042.29; FTDD §11.2). Sourced, not executed.
#
# ONE function, read by both places that name an image's repository in docker-images.yml: the change detector
# (`detect`'s `repo_for`) and the build (`img` step's `repos` output).
#
# From P1 to P7 every image went to two repositories (the old `hitsai/mistudio-<x>` beside `hitsai/miworkbench-<x>`).
# Since P7 (FTASKS 18.10) only the new one receives pushes; the old repositories keep their images and are never
# deleted (DR-27: Docker Hub cannot rename a repository). Pinned by backend/tests/unit/test_image_pushes.py.
image_repos() {
  case "$1" in
    backend)     echo "hitsai/miworkbench-backend" ;;
    frontend)    echo "hitsai/miworkbench-frontend" ;;
    rule-runner) echo "hitsai/miworkbench-rule-runner" ;;
    *)           echo "" ;;
  esac
}
