# Notes

## TODOs

- [x] remove 'texture' label
- [x] add 'pattern shape' label
- [x] drop password
- [x] add lab
- [ ] add 'odd one out' leaderboard
  - per user/lab
  - per challenge
- [ ] add 'detect ai' leaderboard
- [ ] swap login and creat account tabs
- [ ] add comparison task
  - bounding boxes for comparison and ai eval



### free port

```shell
lsof -i :8888 | grep python | awk 'NR==1 {print $2}' | xargs kill -9 && streamlit run app.py --server.port 8888
```
