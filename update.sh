# finally merge all to master branch

# git pull origin master

git add --all
# git rm --cached update.sh
# git rm --cached .DS_Store
# git rm --cached */.DS_Store
# git rm --cached */*/.DS_Store
# git rm --cached */*/*/.DS_Store
# git status
git commit -m "$1"
git push origin HEAD:master

