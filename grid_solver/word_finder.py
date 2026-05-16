from grid_solver.trie import Trie

def load_words(min_len=4):
    words = set()
    with open("words.txt") as f:
        for w in f:
            w = w.strip().upper()
            if len(w) >= min_len:
                words.add(w)
    return words

class WordFinder:
    def __init__(self, grid, word_list):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.trie = Trie()

        for word in word_list:
            self.trie.insert(word)

        self.results = set()

    def find_words(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c]:
                    self.dfs(r, c, self.trie.root, "", set())

        return self.results

    def dfs(self, r, c, node, path, visited):
        if (r < 0 or r >= self.rows or
                c < 0 or c >= self.cols or
                (r, c) in visited):
            return

        letter = self.grid[r][c]
        if letter is None:
            return

        if letter not in node.children:
            return

        visited.add((r, c))
        node = node.children[letter]
        path += letter

        if node.is_word:
            self.results.add(path)

        # 8 directions
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr != 0 or dc != 0:
                    self.dfs(r + dr, c + dc, node, path, visited)

        visited.remove((r, c))