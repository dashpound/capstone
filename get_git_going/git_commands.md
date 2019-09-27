# Git Commands

Know what you're doing?

Here is the cheat commands for reference.

**Clone**
```bash
$ git clone https://github.com/dashpound/capstone.git
```
**Switch to working branch**
```bash
$ git branch jk_dev
```
Where `jk_dev` is arbitrary branch name.

**Commit your work**

```bash 
$ git add --all
$ git status
$ git commit -m"JK: Initial Commit"
$ git status
```
`git add --all` stages your changes

`git status` shows you the files that you've changed

`git commit -m" "` This is your commit message, it should be descriptive.  The format I like is "JK (initials): Message"

**Retrieve changes from remote master repository** 

```bash 
$ git fetch remote master
```


**Integrate your changes** 

```bash
$ git fetch origin
$ git rebase <branch>
$ git push origin branch
```

