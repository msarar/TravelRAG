"""Shared helpers used by every stage (stage1/, stage2/, ...).

Rule of thumb: code goes here only if it does NOT depend on a stage's settings.
Anything a stage might want to change (cleaning output, chunking, paths, models)
is either passed in as an argument or lives in that stage's own folder.
"""
