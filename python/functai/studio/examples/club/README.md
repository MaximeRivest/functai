# Club inbox: a FunctAI Studio example

A robotics club gets messages: questions, ideas and spam. `club.py` sorts each one
(`kind`), answers questions from `club_faq.md` (`answer`), and writes a short reply
(`reply`). `examples/club_inbox.csv` holds twelve messages to try it on.

```bash
functai studio run      # run the twelve examples (it says what it costs first)
functai studio open     # see, judge and fix it in the dashboard
```

Things to find: the FAQ says nothing about fees, so one answer can't be right
whatever the AI does; and "can my little brother join?" may be sorted wrongly.
