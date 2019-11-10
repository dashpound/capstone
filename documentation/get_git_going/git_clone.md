# Git Clone

Git clone is what it is called when you make a working copy of the project code base.

The official documentation is here:
https://git-scm.com/docs/git-clone

You can clone via `ssh` or `https`.  I'm going to move forward assuming `https`.


## The Basics

cd to where you would like to make a copy of the working directory
and execute the git clone command.

Git clone makes a copy of the default branch to the working directory.

```bash
$ cd <filepath>
$ git clone https://github.com/dashpound/capstone
```

I've set the remote origin master to the github repository.

```bash 
$ git remote add origin https://github.com/dashpound/capstone.git
```
