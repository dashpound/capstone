# Git branches

Git branching is a little bit complex but it is important.  How the branches are designed will impact how the work is done.

The official documentation is here:
https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging

## The Basics

The key branch that is protected is called the "Master" branch.  This is a standard convention throughout the programming universe.  

**Remember, NEVER COMMIT TO MASTER** 

Instead, best practice is to create a branch which is a copy of a code at a point in time.  When you select a branch you "checkout a branch" 

To checkout a NEW branch use the command 

```bash
$ git checkout -b <branchname>
```

The -b is only necessary because we're creating a new branch.  

To switch between branches

```
$ git checkout <branchname> 
```

This branch will then diverge as necessary... updates will be made and you'll add features.  Ultimately, you'll want to reintegrate this back to the Master branch.  

There are different strategies for reintegrating to the Master branch including `Merge` and `Rebase`.

I am most familiar with the `Rebase ` methodology but would like to learn how to `Merge` as it is more common in software development.