#include <iostream>
#include <vector>
#include <list>
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <stack>
#include <float.h>
#include <set>
#include <map>

using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

struct ListNode {
    int val;
    ListNode *next;

    ListNode() : val(0), next(nullptr) {}

    ListNode(int x) : val(x), next(nullptr) {}

    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

#define sn struct node

class LRUCache {
public:
    struct node {
        int val;
        int key;
        struct node *f;
        struct node *p;
    };
    struct node *Head;
    struct node *Tail;

    void addNode(sn *current) {
        current->f = Head->f;
        current->f->p = current;
        current->p = Head;
        Head->f = current;
    }

    void removeNode(sn *current) {
        sn *forw = current->f;
        sn *prev = current->p;

        prev->f = forw;
        forw->p = prev;
    }

    sn *popTail() {
        sn *temp = Tail->p;
        removeNode(temp);
        return temp;
    }

    void moveToTop(sn *temp) {
        removeNode(temp);
        addNode(temp);
    }


    unordered_map<int, struct node *> m;
    int sz;
    int filled;

    LRUCache(int capacity) {
        ios_base::sync_with_stdio(false);
        cin.tie(nullptr);
        cout.tie(nullptr);
        sz = capacity;
        filled = 0;
        Head = new node();
        Tail = new node();
        Head->f = Tail;
        Tail->p = Head;
    }

    int get(int key) {
        sn *nodep = m[key];
        if (nodep == NULL) {
            return -1;
        } else {
            moveToTop(nodep);
            return nodep->val;
        }
    }

    void put(int key, int value) {
        sn *nodep = m[key];
        if (nodep == NULL) {
            nodep = new node();
            nodep->val = value;
            nodep->key = key;

            addNode(nodep);
            m[key] = nodep;
            filled++;
            if (filled > sz) {
                sn *temp = popTail();
                m[temp->key] = NULL;
                filled--;
            }
        } else {
            nodep->val = value;
            moveToTop(nodep);
        }
    }
};

class segmentTree {
public:
    int sum, start, end;
    segmentTree *left, *right;

    segmentTree(int start, int end) {
        this->start = start;
        this->end = end;
        this->left = nullptr;
        this->right = nullptr;
        this->sum = 0;
    }
};

class NumArray {
public:
    segmentTree *root = nullptr;

    segmentTree *buildTree(vector<int> &nums, int s, int e) {
        if (s > e) {
            return nullptr;
        } else {
            //for range s and e!
            segmentTree *ret = new segmentTree(s, e);
            if (s == e) {
                ret->sum = nums[s];
            } else {
                int mid = s + (e - s) / 2;
                ret->left = buildTree(nums, s, mid);
                ret->right = buildTree(nums, mid + 1, e);
                //this is the condition where we can change, if we want to evaluate gcd, then instead of sum, we can evaluate gcd of left and right!
                ret->sum = ret->left->sum + ret->right->sum;
            }
            return ret;
        }
    }

    NumArray(vector<int> &nums) {
        root = buildTree(nums, 0, nums.size() - 1);
    }

    void update(int i, int val) {
        update(root, i, val);
    }

    void update(segmentTree *root, int pos, int val) {
        if (root->start == root->end) {
            root->sum = val;
        } else {
            int mid = root->start + (root->end - root->start) / 2;
            if (pos <= mid) {
                update(root->left, pos, val);
            } else {
                update(root->right, pos, val);
            }
            //this is the condition where we can change, if we want to evaluate gcd, then instead of sum, we can evaluate gcd of left and right!
            root->sum = root->left->sum + root->right->sum;
        }
    }

    int sumRange(int i, int j) {
        return sumRange(root, i, j);
    }

    int sumRange(segmentTree *root, int start, int end) {
        if (root->end == end && root->start == start) {
            return root->sum;
        } else {
            int mid = root->start + (root->end - root->start) / 2;
            if (end <= mid) {
                return sumRange(root->left, start, end);
            } else if (start >= mid + 1) {
                return sumRange(root->right, start, end);
            } else {
                return sumRange(root->right, mid + 1, end) + sumRange(root->left, start, mid);
            }
        }
    }
};


class Node {
public:
    int val;
    vector<Node *> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node *> _children) {
        val = _val;
        children = _children;
    }
};


class Codec {
public:
    // Encodes an n-ary tree to a binary tree.
    TreeNode *encode(Node *root) {
        if (root == nullptr)
            return nullptr;
        TreeNode *res = new TreeNode(root->val);
        if (!root->children.empty()) {
            res->left = encode(root->children[0]);
            TreeNode *cur = res->left;
            for (int i = 1; i < root->children.size(); i++) {
                cur->right = encode(root->children[i]);
                cur = cur->right;
            }
        }
        return res;
    }

    // Decodes your binary tree to an n-ary tree.
    Node *decode(TreeNode *root) {
        if (root == nullptr)
            return nullptr;
        Node *res = new Node(root->val);
        TreeNode *sub = root->left;
        while (sub != nullptr) {
            res->children.push_back(decode(sub));
            sub = sub->right;
        }
        return res;
    }
};

class Solution {
public:
    vector<int> rearrangeBarcodes(vector<int> &B) {
        vector<int> count(10001);
        for (auto &b:B)
            count[b]++;
        int freq = 0, val;
        for (int i = 1; i <= 10000; i++)
            if (count[i] > freq) {
                freq = count[i];
                val = i;
            }
        vector<int> res(B.size());
        int idx = 0;
        while (count[val] > 0) {
            res[idx] = val;
            idx += 2;
            if (idx >= B.size())
                idx = 1;
            count[val]--;
        }
        int remain = B.size() - freq, reach = 1;
        while (remain > 0) {
            while (count[reach] > 0) {
                res[idx] = reach;
                idx += 2;
                remain--;
                if (idx >= B.size())
                    idx = 1;
                count[reach]--;
            }
            reach++;
        }
        return res;
    }

    vector<int> findClosestElements(vector<int> &arr, int k, int x) {
        int start = 0, end = arr.size() - k, mid;
        while (start <= end) {
            mid = (start + end) >> 1;
            if (x - arr[mid] >= arr[mid + k - 1] - x)
                start = mid + 1;
            else
                end = mid - 1;
        }
        if (start + k - 1 >= arr.size() || (start > 0 && x - arr[start - 1] <= arr[start + k - 1] - x))
            start--;
        return vector<int>(arr.begin() + start, arr.begin() + start + k);
    }

    vector<int> closestKValues(TreeNode *root, double target, int k) {
        list<int> ls;
        CKHelper(ls, root, target, k);
        vector<int> res;
        for (auto &data:ls)
            res.push_back(data);
        return res;
    }

    void CKHelper(list<int> &ls, TreeNode *root, double target, int k) {
        if (root == nullptr)
            return;
        CKHelper(ls, root->left, target, k);
        if (ls.size() == k) {
            if (target - ls.front() <= root->val - target)
                return;
            ls.pop_front();
        }
        ls.push_back(root->val);
        CKHelper(ls, root->right, target, k);
    }

    int strangePrinter(string s) {
        if (s.empty())
            return 0;
        int N = s.size();
        int dp[N][N];
        for (int i = 0; i < N; i++) {
            dp[i][i] = 1;
            if (i != N - 1) {
                dp[i][i + 1] = s[i] == s[i + 1] ? 1 : 2;
            }
        }
        for (int len = 3; len <= N; len++)
            for (int start = 0; start <= N - len; ++start) {
                int end = start + len - 1;
                dp[start][end] = len;
                for (int k = start; k < end; k++) {
                    int temp = dp[start][k] + dp[k + 1][end];
                    dp[start][end] = min(dp[start][end], s[k] == s[end] ? temp - 1 : temp);
                }
            }
        return dp[0][N - 1];
    }

    template<typename T, int count>
    void foo(T x) {
        T val[count];
        for (int i = 0; i < count; ++i) {
            val[i] = x++;
            cout << val[i] << " ";
        }
    }

    int maxProduct(vector<int> &nums) {
        if (nums.empty())
            return 0;
        int maxPos = 0, maxNeg = 0, res = 0;
        for (int &n:nums) {
            if (n == 0) {
                maxPos = maxNeg = 0;
            } else if (n < 0) {
                int temp = maxPos;
                maxPos = maxNeg * n;
                if (temp != 0)
                    maxNeg = temp * n;
                else
                    maxNeg = n;
            } else {
                maxNeg *= n;
                if (maxPos != 0)
                    maxPos *= n;
                else
                    maxPos = n;
            }
            res = max(res, maxPos);
        }
        return res;
    }

    int minCost(vector<vector<int>> &costs) {
        if (costs.empty())
            return 0;
        vector<vector<int>> dp(costs.size(), vector<int>(3, 0));
        return min(costs[0][0] + MCHelper(1, 0, costs, dp),
                   min(costs[0][1] + MCHelper(1, 1, costs, dp),
                       costs[0][2] + MCHelper(1, 2, costs, dp)));
    }

    int MCHelper(int idx, int ban, vector<vector<int>> &costs, vector<vector<int>> dp) {
        int N = dp.size();
        if (idx >= N)
            return 0;
        if (dp[idx][ban] != 0)
            return dp[idx][ban];
        dp[idx][ban] = INT_MAX;
        for (int i = 0; i < 3; i++) {
            if (i != ban) {
                dp[idx][ban] = min(dp[idx][ban], costs[idx][i] + MCHelper(idx + 1, i, costs, dp));
            }
        }
        return dp[idx][ban];
    }

    int countBinarySubstrings(string s) {
        if (s.empty())
            return 0;
        int cur = 1, pre = 0, res = 0;
        for (int i = 1; i < s.size(); i++) {
            if (s[i] == s[i - 1])
                cur++;
            else {
                pre = cur;
                cur = 1;
            }
            if (cur <= pre)
                res++;
        }
        return res;
    }

    int shortestDistance(vector<vector<int>> &grid) {
        if (grid.empty() || grid[0].empty())
            return 0;
        int R = grid.size(), C = grid[0].size(), build = 0;
        vector<vector<int>> dist(R, vector<int>(C, 0)), reach(R, vector<int>(C, 0));
        int dirs[][2] = {{1,  0},
                         {-1, 0},
                         {0,  1},
                         {0,  -1}};
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) {
                if (grid[i][j] == 1) {
                    SDHelper(i, j, grid, dist, reach, dirs);
                    build++;
                }
            }
        }
        int res = INT_MAX;
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                if (grid[r][c] == 0 && reach[r][c] == build) {
                    res = min(res, dist[r][c]);
                }
            }
        }
        return res == INT_MAX ? -1 : res;
    }

    void SDHelper(int r, int c, vector<vector<int>> &grid, vector<vector<int>> &dist, vector<vector<int>> &reach,
                  int dirs[][2]) {
        queue<pair<int, int>> q;
        q.push({r, c});
        vector<vector<bool>> visited(grid.size(), vector<bool>(grid[0].size(), false));
        int step = 0, R = grid.size(), C = grid[0].size();
        while (!q.empty()) {
            ++step;
            for (int size = q.size(); size > 0; --size) {
                pair<int, int> cur = q.front();
                q.pop();
                for (int i = 0; i < 4; i++) {
                    int nextR = cur.first + dirs[i][0], nextC = cur.second + dirs[i][1];
                    if (nextR < 0 || nextR >= R || nextC < 0 || nextC >= C || visited[nextR][nextC] ||
                        grid[nextR][nextC] != 0)
                        continue;
                    visited[nextR][nextC] = true;
                    q.push({nextR, nextC});
                    reach[nextR][nextC]++;
                    dist[nextR][nextC] += step;
                }
            }
        }
    }

    int minCost(int n, vector<int> &cuts) {
        if (cuts.empty())
            return 0;
        vector<vector<int>> dp(cuts.size(), vector<int>(cuts.size(), 0));
        sort(cuts.begin(), cuts.end());
        return MCHelper(0, cuts.size() - 1, 0, n, cuts, dp);
    }

    int MCHelper(int start, int end, int nLeft, int nRight, vector<int> &cuts, vector<vector<int>> &dp) {
        if (start > end)
            return 0;
        if (dp[start][end] != 0)
            return dp[start][end];
        int temp = INT_MAX;
        for (int i = start; i <= end; i++) {
            temp = min(temp, MCHelper(start, i - 1, nLeft, cuts[i], cuts, dp) +
                             MCHelper(i + 1, end, cuts[i], nRight, cuts, dp));
        }
        temp += nRight - nLeft;
        dp[start][end] = temp;
        return temp;
    }

    int longestOnes(vector<int> &A, int K) {
        if (A.empty())
            return 0;
        int res = 0;
        for (int start = 0, end = 0; end < A.size(); end++) {
            if (A[end] == 0) {
                K--;
                while (K < 0) {
                    if (A[start++] == 0)
                        K++;
                }
            }
            res = max(res, end - start + 1);
        }
        return res;
    }

    int palindromePartition1(string s, int k) {
        if (k >= s.size())
            return 0;
        int N = s.size();
        vector<vector<int>> dp(N, vector<int>(N, 0)), pal(N, vector<int>(N, 0));
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                pal[i][j] = PPHelper(s, i, j);
            }
        }
        for (int i = 0; i < N; i++)
            dp[0][i] = pal[0][i];
        for (int curK = 1; curK < k; ++curK) {
            for (int end = curK - 1; end < N; ++end) {
                int m = N;
                for (int start = end - 1; start >= 0; --start) {
                    m = min(m, dp[curK - 1][start] + pal[start + 1][end]);
                }
                dp[curK][end] = m;
            }
        }
        return dp[k - 1][N - 1];
    }

    int PPHelper(const string &s, int start, int end) {
        int sum = 0;
        while (start < end) {
            if (s[start] != s[end])
                sum++;
            start++;
            end--;
        }
        return sum;
    }

    long POHelper(int start, int remainP, int remainG, vector<vector<vector<long>>> &dp) {
        if (remainP == 0 && remainG == 0)
            return 1;
        if (remainP == 0 || remainG == 0 || start > remainP || remainP < remainG)
            return 0;
        if (dp[start][remainP][remainG] != 0)
            return dp[start][remainP][remainG];
        long res = 0;
        for (int i = start; i <= remainP; ++i) {
            res += POHelper(i, remainP - i, remainG - 1, dp);
        }
        dp[start][remainP][remainG] = res;
        return res;
    }

    long countOptions1(int people, int groups) {
        if (groups > people)
            return 0;
        if (groups == people)
            return 1;
        vector<vector<vector<long>>> dp(people + 1, vector<vector<long>>(people + 1, vector<long>(groups + 1, 0)));
        return POHelper(1, people, groups, dp);
    }

    long minTime1(vector<int> &batchSize, vector<int> &processTime, vector<int> &numTasks) {
        int N = batchSize.size();
        if (N == 0)
            return 0;
        int res = 0;
        for (int i = 0; i < N; ++i) {
            int times = numTasks[i] / batchSize[i] + (numTasks[i] / batchSize[i] != 0);
            res = max(res, times * processTime[i]);
        }
        return res;
    }

    int numSubarrayProductLessThanK(vector<int> &nums, int k) {
        if (k == 0 || nums.empty())
            return 0;
        ios::sync_with_stdio(false);
        cin.tie(0);
        cout.tie(0);
        int res = 0, val = 1;
        for (int start = 0, end = 0; end < nums.size(); end++) {
            val *= nums[end];
            while (val >= k)
                val /= nums[start++];
            res += end - start + 1;
        }
        return res;
    }

    int calculateTotalRotorConfiguration(int n, int start, int end) {
        int mod = pow(10, 9) + 7;
        int range = end - start + 1;
        long long res = 0;
        vector<vector<bool>> coprime(range, vector<bool>(range, false));
        for (int i = 0; i <= range; i++)
            for (int j = i; j <= range; j++) {
                coprime[i + start][j + start] = gcd(i, j) == 1;
            }
        vector<int> count(range, 0);
        for (int i = 0; i <= range; i++)
            for (int j = i; j <= range; j++) {
                if (coprime[i + start][j + start]) {
                    count[i + start]++;
                    count[j + start]++;
                }
            }
        for (int i = 0; i <= range; ++i) {
            res = (res + static_cast<long long>(pow(count[i + start], range - 1))) % mod;
        }
        return static_cast<int>(res);
    }

    int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    string formString(string s) {
        if (s.empty())
            return s;
        vector<int> count(26, 0);
        for (char &c:s)
            count[c - 'a']++;
        string res;
        while (res.size() != s.size()) {
            int start = 0, end = 25;
            while (start != 26) {
                if (count[start] != 0) {
                    res += static_cast<char>(start + 'a');
                    count[start]--;
                }
                start++;
            }
            int last = static_cast<int>(res[res.size() - 1] - 'a');
            while (end >= 0) {
                if (count[end] != 0 && end < last) {
                    res += static_cast<char>(end + 'a');
                    count[end]--;
                }
                end--;
            }
        }
        return res;
    }

    int strokeRequired(vector<string> &data) {
        if (data.empty())
            return 0;
        int R = data.size(), C = data[0].size(), res = 0;
        vector<vector<bool>> visited(R, vector<bool>(C, false));
        vector<vector<int>> dirs{{1,  0},
                                 {-1, 0},
                                 {0,  1},
                                 {0,  -1}};
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                if (!visited[r][c]) {
                    SRHelper(r, c, data, data[r][c], visited, dirs);
                    res++;
                }
        return res;
    }

    void SRHelper(int r, int c, vector<string> &data, char source, vector<vector<bool>> &visited,
                  vector<vector<int>> &dirs) {
        visited[r][c] = true;
        int R = data.size(), C = data[0].size();
        for (vector<int> &d:dirs) {
            int nextR = r + d[0], nextC = c + d[1];
            if (nextR < 0 || nextR >= R || nextC < 0 || nextC >= C || visited[nextR][nextC] ||
                data[nextR][nextC] != source)
                continue;
            SRHelper(nextR, nextC, data, source, visited, dirs);
        }
    }

    int maxMin(vector<int> arr, int k) {
        int res = 0;
        deque<int> dq;
        for (int i = 0; i < arr.size(); i++) {
            if (!dq.empty() && i - dq.front() >= k)
                dq.pop_front();
            while (!dq.empty() && arr[i] <= arr[dq.back()])
                dq.pop_back();
            dq.push_back(i);
            if (i >= k - 1)
                res = max(res, arr[dq.front()]);
        }
        return res;
    }

    long minTime(vector<int> batchSize, vector<int> processTime, vector<int> numTasks) {
        int N = batchSize.size();
        if (N == 0)
            return 0;
        long long res = 0;
        for (int i = 0; i < N; ++i) {
            long long times = numTasks[i] / batchSize[i] + (numTasks[i] % batchSize[i] != 0);
            res = max(res, times * processTime[i]);
        }
        return res;
    }

    long countOptions(int people, int groups) {
        if (people < groups)
            return 0;
        vector<vector<long long>> dp(people + 1, vector<long long>(groups + 1));
        for (int i = 1; i <= groups; ++i) {
            dp[i][i] = 1;
        }

        for (int i = 2; i <= people; ++i) {
            int maxGroup = min(i, groups);
            for (int j = 1; j <= maxGroup; ++j) {
                dp[i][j] = dp[i - 1][j - 1] + dp[i - j][j];
            }
        }
        return dp[people][groups];
    }

    int removeStones(vector<vector<int>> &stones) {
        if (stones.size() <= 1)
            return 0;
        unordered_map<int, int> rows, cols;
        vector<int> id(stones.size()), weight(stones.size());
        for (int i = 0; i < stones.size(); ++i) {
            id[i] = i;
            weight[i] = 1;
        }
        int res = 0;
        for (int i = 0; i < stones.size(); ++i) {
            auto &s = stones[i];
            auto rowIt = rows.find(s[0]), colIt = cols.find(s[1]);
            if (rowIt == rows.end()) {
                rows[s[0]] = i;
            } else if (rsFind(i, id) != rsFind(rows[s[0]], id)) {
                rsUnion(rows[s[0]], i, id, weight);
                res++;
            }
            if (colIt == cols.end()) {
                cols[s[1]] = i;
            } else if (rsFind(i, id) != rsFind(cols[s[1]], id)) {
                rsUnion(cols[s[1]], i, id, weight);
                res++;
            }
        }
        return res;
    }

    int rsFind(int i, vector<int> &id) {
        while (id[i] != i) {
            id[i] = id[id[i]];
            i = id[i];
        }
        return i;
    }

    void rsUnion(int i, int j, vector<int> &id, vector<int> &weight) {
        int idI = rsFind(i, id), idJ = rsFind(j, id);
        if (idI == idJ)
            return;
        if (weight[idI] >= weight[idJ]) {
            id[idJ] = idI;
            weight[idI] += weight[idJ];
        } else {
            id[idI] = idJ;
            weight[idJ] += weight[idI];
        }
    }

    //1-2-3-4-5-6-null
    TreeNode *sortedListToBST(ListNode *head) {
        if (head == nullptr)
            return nullptr;
        if (head->next == nullptr) {
            return new TreeNode(head->val);
        }
        ListNode *slow = head, *fast = head, *prev = nullptr;
        while (fast != nullptr) {
            fast = fast->next;
            if (fast == nullptr)
                break;
            fast = fast->next;
            prev = slow;
            slow = slow->next;
        }
        TreeNode *root = new TreeNode(slow->val);
        if (prev != nullptr)
            prev->next = nullptr;
        root->left = sortedListToBST(head);
        root->right = sortedListToBST(slow->next);
        return root;
    }

    int numSubarrayProductLessThanK2(vector<int> &nums, int k) {
        if (k == 0)
            return 0;
        int res = 0;
        for (int product = 1, start = 0, end = 0; end < nums.size(); end++) {
            product *= nums[end];
            while (product >= k && start <= end) {
                product /= nums[start++];
            }
            res += end - start + 1;
        }
        return res;
    }

    struct comparator {


        bool operator()(const vector<int> &a, const vector<int> &b) {
            //comparison code here
            return a[0] + a[0] - b[0] - b[1] <= 0;
        }
    };

    vector<vector<int>> kSmallestPairs(vector<int> &nums1, vector<int> &nums2, int k) {
        vector<vector<int>> res;
        if (nums1.empty() || nums2.empty() || k <= 0)
            return res;
        auto comp = [&nums1, &nums2](vector<int> a, vector<int> b) {
            return nums1[a[0]] + nums2[a[1]] > nums1[b[0]] + nums2[b[1]];
        };
        priority_queue<vector<int>, vector<vector<int>>, decltype(comp)> pq(comp);
        pq.push({0, 0});
        while (k-- > 0 && !pq.empty()) {
            auto cur = pq.top();
            pq.pop();
            res.push_back({nums1[cur[0]], nums2[cur[1]]});
            if (cur[0] + 1 < nums1.size())
                pq.push({cur[0] + 1, cur[1]});
            if (cur[0] == 0 && cur[1] + 1 < nums2.size())
                pq.push({cur[0], cur[1] + 1});
        }
        return res;
    }

    double findMedianSortedArrays(vector<int> &nums1, vector<int> &nums2) {
        if (nums2.size() < nums1.size()) {
            vector<int> temp = nums1;
            nums1 = nums2;
            nums2 = temp;
        }
        int start = 0, end = nums1.size() - 1, before = (nums1.size() + nums2.size() + 1) >> 1;
        while (start <= end) {
            int id1 = (start + end) >> 1;
            int id2 = before - id1;
            if (id1 != nums1.size() && nums2[id2 - 1] > nums1[id1])
                start = id1 + 1;
            else if (id1 != 0 && nums1[id1 - 1] > nums2[id2])
                end = id1 - 1;
            else {
                int maxL = id1 == 0 ? nums2[id2 - 1] : id2 == 0 ? nums1[id1 - 1] : max(nums1[id1 - 1], nums2[id2 - 1]);
                if (((nums1.size() + nums2.size()) & 1) == 1)
                    return (double) maxL;
                int minR = id1 == nums1.size() ? nums2[id2] : id2 == nums2.size() ? nums1[id1] : min(nums1[id1],
                                                                                                     nums2[id2]);
                return ((double) maxL + (double) minR) / 2;
            }
        }
        return 0;
    }

    int removeBoxes(vector<int> &boxes) {
        if (boxes.empty())
            return 0;
        int n = boxes.size();
        vector<vector<vector<int>>> dp(n, vector<vector<int>>(n, vector<int>(n, 0)));
        return RMHeloer(boxes, dp, 0, n - 1, 0);
    }

    int RMHeloer(vector<int> &B, vector<vector<vector<int>>> &dp, int from, int to, int len) {
        if (from > to)
            return 0;
        if (from == to)
            return (len + 1) * (len + 1);
        if (dp[from][to][len])
            return dp[from][to][len];
        for (; from + 1 <= to && B[from + 1] == B[from]; from++, len++);
        int res = (len + 1) * (len + 1) + RMHeloer(B, dp, from + 1, to, 0);
        for (int i = from + 1; i <= to; i++) {
            if (B[i] == B[from]) {
                res = max(res, RMHeloer(B, dp, from + 1, i - 1, 0) + RMHeloer(B, dp, i, to, len + 1));
            }
        }
        dp[from][to][len] = res;
        return res;
    }

    int maxKilledEnemies(vector<vector<char>> &grid) {
        if (grid.empty() || grid[0].empty())
            return 0;
        int R = grid.size(), C = grid[0].size();
        vector<vector<int>> LR(R + 2, vector<int>(C + 2, 0)), RL(R + 2, vector<int>(C + 2, 0)),
                TB(R + 2, vector<int>(C + 2, 0)), BT(R + 2, vector<int>(C + 2, 0));
        for (int r = 1; r <= R; ++r)
            for (int c = 1; c <= C; ++c) {
                if (grid[r - 1][c - 1] == 'E') {
                    LR[r][c] = LR[r][c - 1] + 1;
                    TB[r][c] = TB[r - 1][c] + 1;
                } else if (grid[r - 1][c - 1] == '0') {
                    LR[r][c] = LR[r][c - 1];
                    TB[r][c] = TB[r - 1][c];
                }
            }
        for (int r = R; r > 0; --r)
            for (int c = C; c > 0; --c) {
                if (grid[r - 1][c - 1] == 'E') {
                    RL[r][c] = RL[r][c + 1] + 1;
                    BT[r][c] = BT[r + 1][c] + 1;
                } else if (grid[r - 1][c - 1] == '0') {
                    RL[r][c] = RL[r][c + 1];
                    BT[r][c] = BT[r + 1][c];
                }
            }
        int res = 0;
        for (int r = 1; r <= R; ++r) {
            for (int c = 1; c <= C; ++c) {
                if (grid[r - 1][c - 1] == '0') {
                    int cur = LR[r][c] + RL[r][c] + TB[r][c] + BT[r][c];
                    res = max(res, cur);
                }
            }
        }
        return res;
    }

    int minMeetingRooms(vector<vector<int>> &I) {
        if (I.empty())
            return 0;
        auto comp = [](int a, int b) {
            return a >= b;
        };
        priority_queue<int, vector<int>, decltype(comp)> pq(comp);
        auto startTimeComp = [](vector<int> &a, vector<int> &b) {
            return a[0] == b[0] ? a[1] < b[1] : a[0] < b[0];
        };
        sort(I.begin(), I.end(), startTimeComp);
        int res = 0;
        for (int i = 0; i < I.size(); ++i) {
            vector<int> cur = I[i];
            if (!pq.empty())
                cout << pq.top() << endl;
            while (!pq.empty() && pq.top() < cur[0]) {
                pq.pop();
            }
            pq.emplace(cur[1]);
            res = max(res, (int) pq.size());
        }
        return res;
    }

    int longestCommonSubsequence(string t1, string t2) {
        if (t1.empty() || t2.empty())
            return 0;
        int N1 = t1.size(), N2 = t2.size();
        vector<vector<int>> dp(N1 + 1, vector<int>(N2 + 1, 0));
        for (int i = 0; i < N1; i++)
            for (int j = 0; j < N2; j++) {
                if (t1[i] == t2[j])
                    dp[i + 1][j + 1] = 1 + dp[i][j];
                else
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1]);
            }
        return dp[N1][N2];
    }

    class NumArray {
    private:
        vector<int> data, ori;
    public:
        NumArray(vector<int> &nums) {
            ori = vector<int>(nums.size(), 0);
            data = vector<int>(nums.size() + 1, 0);
            for (int i = 0; i < nums.size(); i++)
                update(i, nums[i]);
        }

        void update(int i, int val) {
            int diff = val - ori[i];
            for (int idx = i + 1; idx < data.size(); idx += getRange(idx))
                data[idx] += diff;
            ori[i] = val;
        }

        int sumRange(int i, int j) {
            return getSum(j + 1) - getSum(i);
        }

        int getSum(int i) {
            int sum = 0;
            for (int idx = i; idx > 0; idx -= getRange(idx))
                sum += data[idx];
            return sum;
        }

        int getRange(int i) {
            return i & (-i);
        }
    };

    bool isMatch(string s, string p) {
        if (s.empty() && p.empty())
            return true;
        if (p.empty())
            return false;
        int N1 = s.size(), N2 = p.size();
        vector<vector<bool>> dp(N1 + 1, vector<bool>(N2 + 1, false));
        dp[0][0] = true;
        for (int i = 0; i < N2; i++) {
            if (p[i] != '*')
                break;
            dp[0][i + 1] = true;
        }
        for (int i = 0; i < N1; ++i)
            for (int j = 0; j < N2; ++j) {
                if (s[i] == p[j] || p[j] == '?')
                    dp[i + 1][j + 1] = dp[i][j];
                else if (p[j] == '*')
                    dp[i + 1][j + 1] = dp[i + 1][j] || dp[i][j] || dp[i][j + 1];
            }
        return dp[N1][N2];
    }

    int largestRectangleArea(vector<int> &H) {
        if (H.empty())
            return 0;
        int N = H.size(), res = 0;
        vector<int> left(N, -1), right(N, N);
        stack<int> before, after;
        for (int i = 0; i < N; i++) {
            while (!before.empty() && H[before.top()] >= H[i]) {
                before.pop();
            }
            if (!before.empty()) {
                left[i] = before.top();
            }
            before.push(i);

            while (!after.empty() && H[after.top()] > H[i]) {
                right[after.top()] = i;
                after.pop();
            }
            after.push(i);
        }
        for (int i = 0; i < N; i++) {
            if (i != 0 && H[i] == H[i - 1])
                continue;
            int cur = (right[i] - left[i] - 1) * H[i];
            res = max(res, cur);
        }
        return res;
    }

    void rotate(vector<int> &nums, int k) {
        if (nums.empty() || k % nums.size() == 0)
            return;
        k /= nums.size();
        reverseArray(nums, 0, nums.size() - k - 1);
        reverseArray(nums, nums.size() - k, nums.size() - 1);
        reverseArray(nums, 0, nums.size() - 1);
    }

    void reverseArray(vector<int> &nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }

    int palindromePartition(string s, int k) {
        if (s.empty())
            return 0;
        int N = s.size();
//        vector<vector<int>> cost(N,vector<int>(N,0)),dp(k+1,vector<int>(N,0));
        int cost[N][N], dp[k + 1][N];
        memset(cost, 0, sizeof(cost));
        memset(dp, 0, sizeof(dp));
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                cost[i][j] = PPPHelper(s, i, j);
            }
        }
        for (int i = 0; i < N; i++)
            dp[1][i] = cost[i][N - 1];
        for (int curK = 2; curK <= k; curK++)
            for (int i = 0; i < N; i++) {
                int res = INT_MAX;
                for (int j = i + 1; j < N; j++) {
                    res = min(res, cost[i][j] + dp[curK - 1][j]);
                }
                dp[curK][i] = res;
            }
        return dp[k][0];
    }

    inline int PPPHelper(string s, int start, int end) {
        int res = 0;
        while (start < end) {
            if (s[start] != s[end]) {
                res++;
            }
            start++;
            end--;
        }
        return res;
    }

    int countNumbersWithUniqueDigits(int n) {
        if (n == 0)
            return 1;
        int res = 1, cur = 9, curMul = 9;
        for (int i = 0; i < n; ++i) {
            res += cur;
            cur *= curMul;
            curMul--;
        }
        return res;
    }

    int stoneGameII(vector<int> &P) {
        if (P.empty())
            return 0;
        int N = P.size();
        vector<vector<int>> dp(N, vector<int>(N, 0));
        vector<int> sums(N + 1, 0);
        for (int i = N - 1; i >= 0; i--)
            sums[i] = P[i] + sums[i + 1];
        return SGHelper(0, 1, P, dp, sums);
    }

    int SGHelper(int idx, int m, vector<int> &P, vector<vector<int>> &dp, vector<int> &sums) {
        if (idx + (m << 1) >= P.size())
            return sums[idx];
        if (dp[idx][m] != 0)
            return dp[idx][m];
        int res = 0, far = m << 1;
        for (int i = 0; i < far; i++) {
            int curIdx = idx + i;
            int other = SGHelper(idx + i + 1, max(m, i + 1), P, dp, sums);
            int cur = sums[idx] - other;
            res = max(cur, res);
        }
        dp[idx][m] = res;
        return res;
    }

    bool isHappy(int n) {
        int fast = happyNext(n), slow = n;
        while (fast != 1 && fast != slow) {
            fast = happyNext(fast);
            if (fast == 1 || fast == slow)
                break;
            fast = happyNext(fast);
            slow = happyNext(slow);
        }
        return fast == 1;
    }

    inline int happyNext(int n) {
        int res = 0;
        while (n != 0) {
            int temp = n % 10;
            res += temp * temp;
            n /= 10;
        }
        return res;
    }

    int swimInWater(vector<vector<int>> &grid) {
        int N = grid.size(), start = 0, end = N * N - 1;
        while (start <= end) {
            int cur = (start + end) >> 1;
            if (SWisValid(grid, cur))
                end = cur - 1;
            else
                start = cur + 1;
        }
        return start;
    }

    bool SWisValid(const vector<vector<int>> &grid, int depth) {
        int N = grid.size();
        vector<vector<bool>> visited(N, vector<bool>(N, false));
        vector<pair<int, int>> dirs{{1,  0},
                                    {-1, 0},
                                    {0,  1},
                                    {0,  -1}};
        return SWHelper(grid, 0, 0, visited, depth, dirs);
    }

    bool SWHelper(const vector<vector<int>> &grid, int r, int c, vector<vector<bool>> &visited, int depth,
                  const vector<pair<int, int>> &dirs) {
        int N = grid.size();
        if (r < 0 || r >= N || c < 0 || c >= N || grid[r][c] > depth || visited[r][c])
            return false;
        if (r == N - 1 && c == N - 1)
            return true;
        visited[r][c] = true;
        for (int i = 0; i < dirs.size(); ++i) {
            int nextR = r + dirs[i].first, nextC = c + dirs[i].second;
            if (SWHelper(grid, nextR, nextC, visited, depth, dirs))
                return true;
        }
        return false;
    }

    void wiggleSort(vector<int> &nums) {
        if (nums.empty())
            return;
        for (int i = 1; i < nums.size(); i++) {
            if ((((i & 1) == 1) && nums[i] < nums[i - 1]) || (((i & 1) == 0) && nums[i] > nums[i - 1])) {
                swap(nums[i], nums[i - 1]);
//                swap(nums,i,i-1);
            }
        }
    }

    int maxA(int N) {
        if (N <= 0)
            return 0;
        int dp[N + 1];
        memset(dp, 0, sizeof(dp));
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= N; ++i) {
            dp[i] = i;
            for (int j = 3; j <= i; ++j) {
                dp[i] = min(dp[i], dp[j - 2] * (i - j + 1));
            }
        }
        return dp[N];
    }

    int numPairsDivisibleBy60(vector<int> &time) {
        int count[61] = {0};
        int res = 0;
        for (const int &t:time) {
            int div = t % 60;
            res += count[(60 - div) % 60];
            ++count[div];
        }
        return res;
    }

    bool canConvert(string str1, string str2) {
        int count = 0, n = str1.size();
        int map12[26], map21[26];
        memset(map12, -1, sizeof(map12));
        memset(map21, -1, sizeof(map21));
        bool needOpe = false;
        for (int i = 0; i < n; ++i) {
            int idx1 = str1[i] - 'a', idx2 = str2[i] - 'a';
            if (idx1 != idx2) {
                needOpe = true;
            }
            if (map12[idx1] != -1) {
                if (map12[idx1] != idx2)
                    return false;
            } else if (map21[idx2] == -1) {
                ++count;
            }
            map12[idx1] = idx2;
            map21[idx2] = idx1;
        }
        return !needOpe || count < 26;
    }

    int findLongestChain(vector<vector<int>> &pairs) {
        if (pairs.empty()) {
            return 0;
        }
        sort(pairs.begin(), pairs.end(), [](const vector<int> &a, const vector<int> &b) {
            return a[0] == b[0] ? a[1] < b[1] : a[0] < b[0];
        });
        int N = pairs.size();
        int dp[N];
        dp[N - 1] = 1;
        for (int i = N - 2; i >= 0; --i) {
            dp[i] = dp[i + 1];
            int next = FLCHelper(pairs, i + 1, N - 1, pairs[i][1]);
            if (next < N) {
                dp[i] = max(dp[i], 1 + dp[next]);
            }
        }
        return dp[0];
    }

    int FLCHelper(const vector<vector<int>> &P, int start, int end, int min) {
        while (start <= end) {
            int mid = (start + end) >> 1;
            if (P[mid][0] > min) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        return start;
    }

    int maximalSquare(vector<vector<char>> &M) {
        if (M.empty() || M[0].empty()) {
            return 0;
        }
        int R = M.size(), C = M[0].size(), res = 0;
        vector<vector<int>> dp(R + 1, vector<int>(C + 1, 0));
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                if (M[r][c] == '1') {
                    dp[r + 1][c + 1] = min(dp[r][c], min(dp[r + 1][c], dp[r][c + 1])) + 1;
                    res = max(res, dp[r + 1][c + 1]);
                }
            }
        }
        return res * res;
    }

    int subarraysDivByK(vector<int> &A, int K) {
        if (A.empty()) {
            return 0;
        }
        vector<int> count(K, 0);
        count[0] = 1;
        int res = 0, preSum = 0;
        for (const int &a:A) {
            int cur = (a % K + K) % K;
            preSum += cur;
            res += count[preSum % K];
            ++count[preSum % K];
        }
        return res;
    }

    bool increasingTriplet(vector<int> &nums) {
        if (nums.size() < 3) {
            return false;
        }
        int idx1 = -1, idx2 = -1, idx3 = -1;
        for (int i = 0; i < nums.size(); ++i) {
            if (idx1 == -1) {
                idx1 = i;
            } else if (idx2 == -1) {
                if (nums[idx1] > nums[i]) {
                    idx1 = i;
                } else if (nums[i] > nums[idx1]) {
                    idx2 = i;
                }
            } else if (idx3 == -1) {
                if (nums[idx1] > nums[i]) {
                    idx1 = i;
                } else if (nums[idx2] > nums[i] && nums[i] > nums[idx1]) {
                    idx2 = i;
                } else if (nums[i] > nums[idx2]) {
                    idx3 = i;
                    return true;
                }
            }
        }
        return false;
    }

    vector<int> gardenNoAdj(int n, vector<vector<int>> &paths) {
        vector<int> res(n, 0);
        if (n <= 0) {
            return res;
        }
        vector<vector<int>> graph(n + 1, vector<int>());
        for (const vector<int> &p:paths) {
            graph[p[0]].push_back(p[1]);
            graph[p[1]].push_back(p[0]);
        }
        queue<int> q;
        vector<bool> visited(n + 1, false);
        for (int i = 1; i <= n; ++i) {
            if (visited[i]) {
                continue;
            }
            visited[i] = true;
            q.push(i);
            while (!q.empty()) {
                int cur = q.front();
                q.pop();
                int restrict = 0;
                for (const int &adj:graph[cur]) {
                    if (res[adj - 1] != 0) {
                        restrict |= (1 << res[adj - 1]);
                    }
                    if (!visited[adj]) {
                        visited[adj] = true;
                        q.push(adj);
                    }
                }
                for (int j = 1; j <= 4; ++j) {
                    if ((restrict & (1 << j)) == 0) {
                        res[cur - 1] = j;
                        break;
                    }
                }
            }
        }
        return res;
    }

    vector<int> assignBikes1(vector<vector<int>> &W, vector<vector<int>> &B) {
        auto comp = [](const vector<int> &a, const vector<int> &b) { //[dist,wIdx,bIdx]
            return a[0] != b[0] ? a[0] > b[0] : a[1] == b[1] ? a[1] > b[1] : a[2] > b[2];
        };
        priority_queue<vector<int>, vector<vector<int>>, decltype(comp)> pq(comp);
        for (int i = 0; i < W.size(); ++i) {
            for (int j = 0; j < B.size(); ++j) {
                int dist = abs(W[i][0] - B[j][0]) + abs(W[i][1] - B[j][1]);
                pq.push({dist, i, j});
            }
        }
        vector<bool> usedB(B.size(), false), usedW(W.size(), false);
        vector<int> res(W.size(), -1);
        int count = W.size();
        while (count > 0) {
            vector<int> cur = pq.top();
            pq.pop();
            if (usedW[cur[1]] || usedB[cur[2]]) {
                continue;
            }
            res[cur[1]] = cur[2];
            --count;
            usedW[cur[1]] = usedB[cur[2]] = true;
        }
        return res;
    }

    vector<int> assignBikes(vector<vector<int>> &W, vector<vector<int>> &B) {
        vector<vector<vector<int>>> bucket(2001, vector<vector<int>>());
        for (int i = 0; i < W.size(); ++i) {
            for (int j = 0; j < B.size(); ++j) {
                int dist = abs(W[i][0] - B[j][0]) + abs(W[i][1] - B[j][1]);
                bucket[dist].push_back({i, j});
            }
        }
        vector<bool> usedB(B.size(), false);
        vector<int> res(W.size(), -1);
        auto comp = [](const vector<int> &a, const vector<int> &b) { //[dist,wIdx,bIdx]
            return a[0] != b[0] ? a[0] < b[0] : a[1] < b[1];
        };
        for (int idx = 1, unfilled = W.size(); idx < 2001 && unfilled > 0; idx++) {
            if (bucket[idx].empty()) {
                continue;
            }
            sort(bucket[idx].begin(), bucket[idx].end(), comp);
            for (const vector<int> &t:bucket[idx]) {
                if (res[t[0]] != -1 || usedB[t[1]]) {
                    continue;
                }
                --unfilled;
                res[t[0]] = t[1];
                usedB[t[1]] = true;
            }
        }
        return res;
    }

    int canCompleteCircuit(vector<int> &gas, vector<int> &cost) {
        int totalCost = 0, totalGas = 0, N = gas.size();
        for (int i = 0; i < N; ++i) {
            totalCost += cost[i];
            totalGas += gas[i];
        }
        if (totalGas < totalCost) {
            return -1;
        }
        int start = 0, cur = 0;
        for (int i = 0; i < N; ++i) {
            cur += gas[i] - cost[i];
            if (cur < 0) {
                start = i + 1;
                cur = 0;
            }
        }
        return start;
    }

    int knightDialer(int n) {
        int mod = pow(10, 9) + 7;
        vector<vector<int>> jumpTo{{4, 6},
                                   {6, 8},
                                   {7, 9},
                                   {4, 8},
                                   {3, 9, 0},
                                   {},
                                   {1, 7, 0},
                                   {2, 9},
                                   {1, 3},
                                   {2, 4}};
        vector<vector<int>> dp{10, vector<int>(n, 0)};
        for (int i = 0; i < 10; ++i) {
            dp[i][0] = 1;
        }
        for (int time = 1; time < n; ++time) {
            for (int i = 0; i < 10; ++i) {
                for (const int &jump:jumpTo[i]) {
                    dp[i][time] = (dp[i][time] + dp[jump][time - 1]) % mod;
                }
            }
        }
        int res = 0;
        for (int i = 0; i < 10; ++i) {
            res = (res + dp[i][n - 1]) % mod;
        }
        return res;
    }

    vector<vector<int>> merge(vector<vector<int>> &I) {
        if (I.empty()) {
            return I;
        }
        sort(I.begin(), I.end(), [](const vector<int> &a, const vector<int> &b) {
            return a[0] < b[0];
        });
        vector<vector<int>> res;
        for (const vector<int> &i:I) {
            if (res.empty() || res[res.size() - 1][1] < i[0]) {
                res.push_back(i);
            } else {
                res[res.size() - 1][1] = max(res[res.size() - 1][1], i[1]);
            }
        }
        return res;
    }

    template<typename T, int size>
    struct TrieNode {
        T val;
        vector<shared_ptr<TrieNode<T, size>>> child;

        TrieNode(T _val) : val(_val) {
            child = vector<shared_ptr<TrieNode<T, size>>>(size, nullptr);
        }
    };

    void insertTrie(struct TrieNode<bool, 26> *root, const string &str) {
        for (const char &c:str) {
            int idx = c - 'a';
            if (root->child[idx] == nullptr) {
                root->child[idx] = shared_ptr<TrieNode<bool, 26>>(new TrieNode<bool, 26>(false));
            }
            root = root->child[idx].get();
        }
        root->val = true;
    }

    bool wordBreak(string s, vector<string> &wordDict) {
        if (s.empty() || wordDict.empty()) {
            return false;
        }
        TrieNode<bool, 26> root(false);
        for (const string &w:wordDict) {
            insertTrie(&root, w);
        }
        vector<int> memo(s.size() + 1, 0);
        return WBFind(s, 0, &root, memo);
    }

    bool WBFind(const string &s, int idx, TrieNode<bool, 26> *root, vector<int> &memo) {
        if (idx == s.size()) {
            return true;
        }
        if (memo[idx] != 0) {
            return memo[idx] > 0;
        }
        bool res = false;
        TrieNode<bool, 26> *cur = root;
        while (idx < s.size()) {
            int order = s[idx] - 'a';
            if (cur->child[order].get() == nullptr) {
                break;
            }
            if (cur->child[order].get()->val && WBFind(s, idx + 1, root, memo)) {
                res = true;
                break;
            }
            cur = cur->child[order].get();
            ++idx;
        }
        memo[idx] = res ? 1 : -1;
        return res;
    }

    vector<int> splitIntoFibonacci(string S) {
        if (S.size() < 3) {
            return vector<int>();
        }
        int N = S.size();
        vector<int> res;
        for (int i = 0; i < N - 2; i++) {
            long num1 = stol(S.substr(0, i + 1));
            if ((S[0] == '0' && i > 0) || (num1 > INT_MAX)) {
                break;
            }
            res.push_back(num1);
            for (int j = i + 1; j < N - 1; j++) {
                long num2 = stol(S.substr(i + 1, j - i));
                if ((S[i + 1] == '0' && j > i + 1) || (num2 > INT_MAX)) {
                    break;
                }
                res.push_back(num2);
                if (SFHelper(S, j + 1, res)) {
                    return res;
                }
                res.pop_back();
            }
            res.pop_back();
        }
        return res;
    }

    bool SFHelper(const string &s, int start, vector<int> &res) {
        if (start == s.size()) {
            return true;
        }
        long val = (long) res[res.size() - 1] + (long) res[res.size() - 2];
        if (val > INT_MAX) {
            return false;
        }
        string target = to_string(val);
        for (int i = 0; i < target.size(); i++) {
            if (i + start >= s.size() || s[i + start] != target[i]) {
                return false;
            }
        }
        res.push_back(val);
        if (SFHelper(s, start + target.size(), res)) {
            return true;
        }
        res.pop_back();
        return false;
    }

    string getPermutation(int n, int k) {
        vector<int> fact(n + 1);
        vector<bool> used(n + 1, false);
        fact[0] = 1;
        for (int i = 1; i <= n; ++i) {
            fact[i] = fact[i - 1] * i;
        }
        int sum = 0;
        string res;
        for (int i = n; i > 0; --i) {
            for (int j = 1; j <= n; j++) {
                if (used[j]) {
                    continue;
                }
                if (sum + fact[i - 1] >= k) {
                    res += std::to_string(j);
                    used[j] = true;
                    break;
                }
                sum += fact[i - 1];
            }
        }
        return res;
    }

    int shortestPathLength(vector<vector<int>> &graph) {
        int res = 0, N = graph.size(), target = (1 << N) - 1;
        unordered_set<int> visited; // [pos,status]
        queue<pair<int, int>> q;
        for (int i = 0; i < N; ++i) {
            pair<int, int> temp = {i, 1 << i};
            q.push(temp);
            visited.insert(key(temp.first, temp.second));
        }
        while (!q.empty()) {
            res++;
            for (int size = q.size(); size > 0; --size) {
                auto cur = q.front();
                q.pop();
                for (const int &adj:graph[cur.first]) {
                    pair<int, int> next{adj, cur.second | (1 << adj)};
                    if (visited.find(key(next.first, next.second)) != visited.end()) {
                        continue;
                    }
                    if (next.second == target) {
                        return res;
                    }
                    visited.insert(key(next.first, next.second));
                    q.push(next);
                }
            }
        }
        return 0;
    }

    inline int key(int pos, int status) {
        return (pos << 15) | status;
    }

    string findLongestWord(string s, vector<string> &d) {
        if (s.empty() || d.empty()) {
            return nullptr;
        }
        string res;
        for (const string &c:d) {
            if (FLContain(s, c) && (c.size() > res.size() || (c.size() == res.size() && c < res))) {
                res = c;
            }
        }
        return res;
    }

    inline bool FLContain(const string &a, const string &b) {
        if (a.size() < b.size()) {
            return false;
        }
        int idxB = 0;
        for (int idxA = 0; idxA < a.size() && idxB < b.size(); ++idxA) {
            if (a[idxA] == b[idxB]) {
                idxB++;
            }
        }
        return idxB == b.size();
    }

    long long groupingOptions(int n, int m) {
        // write your code here
        //dp[P][T] = dp[P-1][T-1]+ dp[P-T][T]
        if (m > n) {
            return 0;
        }
        if (m == n) {
            return 1;
        }
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
        dp[0][0] = 1;
        for (int p = 1; p <= n; ++p) {
            for (int t = 1; t <= p; ++t) {
                dp[p][t] = dp[p - 1][t - 1] + dp[p - t][t];
            }
        }
        return dp[n][m];
    }

    bool reorderedPowerOf2(int N) {
        string target = RPEncode(N);
        for (unsigned int i = (1 << 31); i > 0; i >>= 1) {
            string cur = RPEncode(i);
            if (cur == target) {
                return true;
            }
        }
        return false;
    }

    inline string RPEncode(unsigned int a) {
        int count[10];
        memset(count, 0, sizeof(count));
        for (; a > 0; a /= 10) {
            ++count[a % 10];
        }
        string res;
        for (int i = 0; i < 10; ++i) {
            for (int j = count[i]; j > 0; --j) {
                res += to_string(i);
            }
        }
        return res;
    }

    int divide(int D, int d) {
        if (D == 0) {
            return 0;
        }
        if (D == INT_MIN && d == -1) {
            return INT_MAX;
        }
        bool isNeg = (D < 0 && d > 0) || (D > 0 && d < 0);
        int res = DVHelper(abs((long) D), abs((long) d));
        return isNeg ? -res : res;
    }

    int DVHelper(long D, long d) {
        if (D < d) {
            return 0;
        }
        long cur = d, count = 0;
        while (D >= cur) {
            cur <<= 1;
            count++;
        }
        long base = d << (count - 1);
        return (1 << (count - 1)) + DVHelper(D - (d << (count - 1)), d);
    }

    int integerReplacement(int n) {
        if (n <= 1) {
            return 0;
        }
        unordered_map<long, int> memo;
        return IRUtil((long) n, memo);
    }

    int IRUtil(long n, unordered_map<long, int> &memo) {
        if (n == 1) {
            return 0;
        }
        if (memo.find(n) != memo.end()) {
            return memo[n];
        }
        int res = (n & 1) == 0 ? 1 + IRUtil(n >> 1, memo) : 1 + min(IRUtil(n + 1, memo), IRUtil(n - 1, memo));
        memo[n] = res;
        return res;
    }

    int maxProfit(int k, vector<int> &prices) {
        if (k == 0) {
            return 0;
        }
        if (prices.size() < 2) {
            return 0;
        }
        int N = prices.size();
        vector<vector<int>> buy(N, vector<int>(k + 1)), sell(N, vector<int>(k + 1));
        for (int t = 1; t <= k; ++t) {
            buy[0][t] = -prices[0];
        }
        for (int time = 1; time <= k; time++) {
            for (int i = 1; i < N; ++i) {
                buy[i][time] = max(buy[i - 1][time], sell[i - 1][time - 1] - prices[i]);
                sell[i][time] = max(sell[i - 1][time], buy[i - 1][time] + prices[i]);
            }
        }
        return sell[N - 1][k];
    }

    vector<double> getCollisionTimes(vector<vector<int>> &cars) {
        int N = cars.size();
        vector<double> res(N, DBL_MAX);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < i; j++) {
                if (cars[i][1] < cars[j][1]) {
                    double curTime =
                            ((double) cars[i][0] - (double) cars[j][0]) / ((double) cars[j][1] - (double) cars[i][1]);
                    res[j] = min(res[j], curTime);
                }
            }
        }
        for (double &r:res) {
            if (r == DBL_MAX) {
                r = -1.0;
            }
        }
        return res;
    }

    int subarrayBitwiseORs(vector<int> &arr) {
        if (arr.size() <= 1) {
            return arr.size();
        }
        unordered_set<int> last, cur, ans;
        for (int &a:arr) {
            cur.clear();
            cur.insert(a);
            for (int r:last) {
                cur.insert(a | r);
            }
            swap(last, cur);
            ans.insert(last.begin(), last.end());
        }
        return ans.size();
    }

    int maxProfit(vector<int> &P) {
        if (P.empty()) {
            return 0;
        }
        int N = P.size();
        vector<vector<int>> buy(3, vector<int>(N, 0));
        vector<vector<int>> sell(3, vector<int>(N, 0));
        buy[1][0] = buy[2][0] = -P[0];
        for (int i = 1; i <= 2; i++) {
            for (int j = 1; j < N; j++) {
                buy[i][j] = max(buy[i][j - 1], sell[i - 1][j - 1] - P[j]);
                sell[i][j] = max(sell[i][j - 1], buy[i][j - 1] + P[j]);
            }
        }
        return max(sell[2][N - 1], sell[1][N - 1]);
    }

    bool isRectangleOverlap(vector<int> &rec1, vector<int> &rec2) {
        if (rec1[0] == rec1[2] || rec1[1] == rec1[3] || rec2[0] == rec2[2] || rec2[1] == rec2[3]) {
            return false;
        }
        return !(rec1[0] >= rec2[2] || rec1[2] <= rec2[0] || rec1[1] >= rec2[3] || rec1[3] <= rec2[1]);
    }

    bool canTransform(string start, string end) {
        if (start.size() != end.size()) {
            return false;
        }
        int N = start.size();
        int idx1 = 0, idx2 = 0;
        for (;; ++idx1, ++idx2) {
            for (; idx1 < N && start[idx1] == 'X'; ++idx1);
            for (; idx2 < N && end[idx2] == 'X'; ++idx2);
            if (idx1 == N || idx2 == N) {
                break;
            }
            if (start[idx1] != end[idx2] || (start[idx1] == 'L' && idx1 < idx2) ||
                (start[idx1] == 'R' && idx1 > idx2)) {
                return false;
            }
        }
        for (; idx1 < N && start[idx1] == 'X'; ++idx1);
        for (; idx2 < N && end[idx2] == 'X'; ++idx2);
        return idx1 == N && idx2 == N;
    }


    bool checkIfPangram(string sentence) {
        if (sentence.size() < 26) {
            return false;
        }
        bool count[26];
        memset(count, false, sizeof(count));
        int remain = 26;
        for (const char &c:sentence) {
            int idx = static_cast<int>(c - 'a');
            if (!count[idx]) {
                remain--;
                count[idx] = true;
                if (remain == 0) {
                    return true;
                }
            }
        }
        return false;
    }

    int maxIceCream(vector<int> &costs, int coins) {
        int N = costs.size(), minVal = INT_MAX, res = 0;
        int64_t sum = 0;
        vector<int> count(100001, 0);
        for (int c:costs) {
            sum += c;
            minVal = min(minVal, c);
            count[c]++;
        }
        if (coins < minVal) {
            return 0;
        }
        if ((int64_t) coins > sum) {
            return N;
        }
        int64_t cur = 0;
        for (int i = 1; i <= 100000; i++) {
            if (count[i] == 0) {
                continue;
            }
            if (cur + (int64_t) i * (int64_t) count[i] < (int64_t) coins) {
                res += count[i];
                cur += (int64_t) i * (int64_t) count[i];
            } else {
                res += (coins - cur) / i;
                break;
            }
        }
        return res;
    }

    vector<int> getOrder(vector<vector<int>> &tasks) {
        int N = tasks.size(), curIdx = 0, taskIdx = 0;
        int64_t curTime = 0;
        vector<int> res(N);
        typedef pair<int, vector<int>> info;
        auto sortComp = [](const info &a, const info &b) {
            return a.second[0] != b.second[0] ? a.second[0] < b.second[0] : a.first < b.first;
        };
        vector<info> data(N);
        for (int i = 0; i < N; ++i) {
            data[i] = {i, tasks[i]};
        }
        sort(data.begin(), data.end(), sortComp);
        auto comp = [](const info &a, const info &b) {
            return a.second[1] == b.second[1] ? a.first > b.first : a.second[1] > b.second[1];
        };
        priority_queue<info, vector<info>, decltype(comp)> pq(comp);
        while (curIdx < N) {
            if (taskIdx < N && pq.empty() && curTime < data[taskIdx].second[0]) {
                curTime = data[taskIdx].second[0];
            }
            while (taskIdx < N && curTime >= data[taskIdx].second[0]) {
                pq.push(data[taskIdx]);
                taskIdx++;
            }
            auto curTask = pq.top();
            pq.pop();
            res[curIdx++] = curTask.first;
            curTime += (int64_t) curTask.second[1];
        }
        return res;
    }

    //[a,b], [c,d,e]
    //(a&c)^(a&d)^(a&e)^(b&c)^(b&d)^(b&e)
    int getXORSum(vector<int> &arr1, vector<int> &arr2) {
        int resA = 0, resB = 0;
        for (int &a:arr1)
            resA ^= a;
        for (int &a:arr2)
            resB ^= a;
        return resA & resB;
    }

    int jump(vector<int> &nums) {
        int res = 0;
        int to = 0, next = 0;
        for (int i = 0; i < nums.size() - 1 && next < nums.size() - 1;) {
            for (; i < nums.size() - 1 && i <= to; ++i) {
                next = max(next, i + nums[i]);
            }
            to = next;
            res++;
        }
        return res;
    }

    int maximumSum(vector<int> &arr) {
        if (arr.empty()) {
            return 0;
        }
        if (arr.size() == 1) {
            return arr[0];
        }
        int N = arr.size(), res = INT_MIN;
        vector<int> start(N + 2), end(N + 2);
        for (int i = 0; i < N; i++) {
            start[i + 1] = start[i] > 0 ? start[i] + arr[i] : arr[i];
        }
        for (int i = N - 1; i >= 0; i--) {
            end[i + 1] = end[i + 2] > 0 ? end[i + 2] + arr[i] : arr[i];
        }
        for (int i = 0; i < N; i++) {
            res = max(res, max(start[i + 1], start[i] + end[i + 2]));
        }
        return res;
    }

    int sumBase(int n, int k) {
        int res = 0;
        while (n > 0) {
            res += n % k;
            n /= k;
        }
        return res;
    }

    int longestBeautifulSubstring(string word) {
        if (word.size() < 5) {
            return 0;
        }
        vector<char> vow{'a', 'e', 'i', 'o', 'u'};
        vector<int> pos(26, 0);
        for (int i = 0; i < vow.size(); i++) {
            pos[vow[i] - 'a'] = i;
        }
        int res = 0;
        for (int start = 0, till = -1, end = 0; end < word.size(); end++) {
            int curPos = pos[word[end] - 'a'];
            if (curPos != till && curPos != (till + 1)) {
                //illegal
                till = curPos ? -1 : 0;
                start = curPos ? end + 1 : end;
            }
            //legal
            if (curPos == till + 1) {
                till++;
            }
            if (till == vow.size() - 1) {
                res = max(res, end - start + 1);
            }
        }
        return res;
    }

    int maxA2(int N) {
        vector<int> dp(N + 1);
        for (int i = 1; i <= N; ++i) {
            dp[i] = i;
            for (int j = 1; j <= i - 3; j++) {
                dp[i] = max(dp[i - j - 2] * (j + 1), dp[i]);
            }
        }
        int temp = dp[N];
        return dp[N];
    }

    int triangleNumber(vector<int> &nums) {
        if (nums.size() < 3) {
            return 0;
        }
        int res = 0;
        sort(nums.begin(), nums.end());
        for (int i = 2; i < nums.size(); ++i) {
            int start = 0;
            for (int j = i - 1; j > start; j--) {
                while (start < j && nums[start] + nums[j] <= nums[i]) {
                    start++;
                }
                res += j - start;
            }
        }
        return res;
    }

    vector<double> calcEquation(vector<vector<string>> &E, vector<double> &V, vector<vector<string>> &Q) {
        vector<double> res(Q.size());
        if (Q.empty()) {
            return res;
        }
        unordered_map<string, unordered_map<string, double>> graph;
        for (int i = 0; i < E.size(); ++i) {
            string from = E[i][0], to = E[i][1];
            double rel = V[i];
            if(graph.count(from)==0){
                unordered_map<string,double> temp;
                graph[from] = temp;
            }
            if(graph.count(to)==0){
                unordered_map<string,double> temp;
                graph[to] = temp;
            }
            graph[from][to]=rel;
            graph[to][from]=1/rel;
        }
        unordered_map<string, double> cache;
        unordered_map<string,int> id;
        unordered_set<string> visited;
        int curId=1;
        for (const auto &cur:graph) {
            if(visited.find(cur.first)==visited.end()){
                CEHelper(cur.first,1,curId++,graph,visited,cache,id);
            }
        }
        for(int i=0;i<Q.size();++i){
            string from = Q[i][0],to = Q[i][1];
            if(cache.count(from)==0 || cache.count(to)==0 || id[from]!=id[to]){
                res[i]=-1;
            }
            else{
                res[i] = cache[Q[i][0]]/cache[Q[i][1]];
            }
        }
        return res;
    }

    void CEHelper(const string &cur,double curVal, int curId, unordered_map<string, unordered_map<string, double>> &graph,
                  unordered_set<string> &visited, unordered_map<string, double> &cache,unordered_map<string,int>& id) {
        visited.insert(cur);
        cache[cur]=curVal;
        id[cur]=curId;
        for(const auto &adj:graph[cur]){
            if(visited.find(adj.first)==visited.end()){
                CEHelper(adj.first,curVal/adj.second,curId,graph,visited,cache,id);
            }
        }
    }

    int maxProduct2(vector<int>& nums) {
        if(nums.empty()){
            return 0;
        }
        if(nums.size()==1){
            return nums[0];
        }
        int minNeg=0,maxPos = 0,res=INT_MIN;
        for(const int &n:nums){
            if (n>=0){
                maxPos = maxPos?maxPos*n:n;
                minNeg = minNeg?minNeg*n:0;
            }
            else{
                int temp = minNeg;
                minNeg = maxPos?maxPos*n:n;
                maxPos = temp*n;
            }
            res = max(res,maxPos);
        }
        return res;
    }

    int maxCoins(vector<int>& nums) {
        if(nums.size()==1){
            return nums[0];
        }
        int N = nums.size();
        vector<vector<int>> dp(N,vector<int>(N,0));
        return MCHelper(nums,dp,0,N-1);
    }

    int MCHelper(const vector<int>& nums,vector<vector<int>>& dp,int start,int end){
        if(start>end){
            return 0;
        }
        int prev = start==0?1:nums[start-1];
        int after = end==nums.size()-1?1:nums[end+1];
        if(start==end){
            return prev*nums[start]*after;
        }
        if(dp[start][end]){
            return dp[start][end];
        }
        int temp = 0;
        for(int i=start;i<=end;++i){
            temp = max(temp,prev*after*nums[i]+MCHelper(nums,dp,start,i-1)+MCHelper(nums,dp,i+1,end));
        }
        dp[start][end] = temp;
        return temp;
    }

    int numTilePossibilities(string T) {
        vector<int> count(26,0);
        for(const char& t:T){
            count[t-'A']++;
        }
        int res=0;
        NTPHelper(count,res);
        return res;
    }

    void NTPHelper(vector<int>& count, int& res){
        for(int i=0;i<26;++i){
            if(!count[i]){
                continue;
            }
            count[i]--;
            res++;
            NTPHelper(count,res);
            count[i]++;
        }
    }

    string crackSafe(int n, int k) {
        string res;
        if (n==1){
            for(int i=0;i<k;i++){
                res+=to_string(i);
            }
            return res;
        }
        if (k==1){
            for(int i=0;i<n;i++){
                res+='0';
            }
            return res;
        }
    }

    int countRangeSum(vector<int>& nums, int lower, int upper) {
        int N = nums.size();
        vector<int64_t> preSum(N+1,0);
        for(int i=0;i<N;++i){
            preSum[i+1]=(int64_t)preSum[i]+(int64_t)nums[i];
        }
        vector<int64_t> aux(N+1);
        return CRSHelper(preSum,aux,0,N,lower,upper);
    }

    int CRSHelper(vector<int64_t>& preSum,vector<int64_t>& aux,int start,int end,int lower,int upper){
        if(start>=end){
            return 0;
        }
        int mid = (start+end)>>1;
        int left =CRSHelper(preSum,aux,start,mid,lower,upper);
        int right = CRSHelper(preSum,aux,mid+1,end,lower,upper);
        return left+right+CRSMerge(preSum,aux,start,end,lower,upper);
    }

    int CRSMerge(vector<int64_t>& preSum,vector<int64_t>& aux,int start,int end,int lower,int upper){
        int mid = (start+end)>>1;
        int res=0;
        int idx = start,RIdx=mid+1,L = mid+1,R = mid+1,auxIdx=start;
        for(;idx<=mid;idx++){
            while (RIdx<=end && preSum[RIdx]<=preSum[idx]){
                aux[auxIdx++]=preSum[RIdx++];
            }
            for(;L<=end && preSum[L]-preSum[idx]<(int64_t)lower;L++);
            for(;R<=end && preSum[R]-preSum[idx]<=(int64_t)upper;R++);
            res+=R-L;
            aux[auxIdx++] = preSum[idx];
        }
        for(;RIdx<=end;){
            aux[auxIdx++] = preSum[RIdx++];
        }
        for(int i=start;i<=end;++i){
            preSum[i] = aux[i];
        }
        return res;
    }

    int getMinDistance(vector<int>& nums, int target, int start) {
        for(int i=start,dist=0;i-dist>=0 || i+dist<nums.size();dist++){
            if(i-dist>=0 && nums[i-dist]==target){
                return dist;
            }
            if(i+dist<nums.size() && nums[i+dist]==target){
                return dist;
            }
        }
        return -1;
    }

    bool splitString(string s) {
        if(s.size()==1){
            return false;
        }
        return SSHelper(0,-1,s)>1;
    }

    int SSHelper(int idx,int64_t last,string& s){
        if(idx==s.size()){
            return 0;
        }
        int64_t cur=0;
        for(int i=idx;i<s.size();i++){
            cur = cur*10 + (uint64_t)(s[i]-'0');
            if(last!=-1 && cur>=last){
                break;
            }
            if(getNumLen(cur)>10){
                break;
            }
            if(last==-1 || cur==last-1){
                int res = SSHelper(i+1,cur,s);
                if(res>=0){
                    return res+1;
                }
            }
        }
        return -1;
    }

    int getNumLen(int64_t num){
        int res=0;
        for(;num;num/=10,res++);
        return res;
    }

    int getMinSwaps(string num, int k) {
        string origin = num;
        GMSgetNext(num,k);
        return GMSMinSwap(origin,num);
    }

    int GMSMinSwap(string& from,string& to){
        int startIdx=0,res=0,N = from.size();
        for(;startIdx<N && from[startIdx]==to[startIdx];++startIdx);
        for(int i=startIdx;i<N;i++){
            bool found = false;
            for(int j=i;j<N && !found;j++){
                if(to[i]==from[j]){
                    for(int k=j;k>i;k--){
                        swap(from[k],from[k-1]);
                        res++;
                    }
                    found=true;
                    break;
                }
            }
        }
        return res;
    }

    void GMSgetNext(string& num,int k){
        auto getNext = [&](){
            for(int i=num.size()-2;i>=0;--i){
                if(num[i]>=num[i+1]){
                    continue;
                }
                int targetIdx = num.size()-1;
                for(;targetIdx>i && num[targetIdx]<=num[i];targetIdx--);
                swap(num[i],num[targetIdx]);
                for(int start = i+1,end = num.size()-1;start<end;){
                    swap(num[start],num[end]);
                    start++;
                    end--;
                }
                break;
            }
        };
        for(int i=0;i<k;++i){
            getNext();
        }
    }

    int countDigitOne(int n) {
        int res=0;
        for(int64_t i=1;i<=n;i*=10){
            int prefix = n/i;
            int after = n%i;
            res += (prefix+8)/10*i+(prefix%10==1)*(after+1);
        }
        return res;
    }

    int findMinDifference(vector<string>& timePoints) {
        int res = 0;
        auto getMins = [](const string& time)->int{
            int hour = (time[0]-'0')*10+(time[1]-'0');
            int min = (time[3]-'0')*10+(time[4]-'0');
            return hour*60+min;
        };
        vector<int> timeSlot(24*60,0);
        int minT= INT_MAX,maxT = INTMAX_MAX;
        for(const string& t:timePoints){
            int curMin = getMins(t);
            minT = min(curMin,minT);
            maxT = max(curMin,maxT);
            timeSlot[curMin]++;
        }
        res = minT+24*60-maxT;
        for(int last = 0,cur=24*60-1;cur>=0 && res>0;cur--){
            if(!timeSlot[cur]){
                continue;
            }
            if(timeSlot[cur]>1){
                return 0;
            }
            if(last){
                res=min(res,last-cur);
            }
            last = cur;
        }
        return res;
    }

    vector<string> removeInvalidParentheses(string s) {
        vector<string> res;
        if(s.empty()){
            return res;
        }
        vector<char> op{'(',')'};
        RIPHelper(s,0,op,res);
        return res;
    }

    //"(((k()(("
    void RIPHelper(string cur,int idx,const vector<char>& op,vector<string>& res){
        int curSum=0;
        cout<<cur<<endl;
        for(int i=idx;i<cur.size();++i){
            if(cur[i]!='(' && cur[i]!=')'){
                continue;
            }
            curSum+= cur[i]==op[0]?1:-1;
            if(curSum<0){
                int next=i;
                for(;next+1<cur.size() && cur[next+1]==op[1];++next);
                if(next!=i){
                    cur = cur.substr(0,i+1)+cur.substr(next+1);
                    cout<<cur<<endl;
                }
                for(int j=0;j<=i;j++){
                    if(cur[j]!=op[1] || (j>0 && cur[j-1]==op[1])){
                        continue;
                    }
                    RIPHelper(cur.substr(0,j)+cur.substr(j+1),i,op,res);
                }
                return;
            }
        }
        reverse(cur.begin(),cur.end());
        if (op[0]=='('){
            vector<char> newOp{')','('};
            RIPHelper(cur,0,newOp,res);
        }
        else{
            res.push_back(cur);
        }
    }

    int monotoneIncreasingDigits(int N) {
        string num = to_string(N);
        int startIdx = -1,minDigit=num[num.size()-1]-'0';
        for(int mask=num.size()-2;mask>=0;mask--){
            int curDigit = num[mask]-'0';
            if(curDigit>minDigit){
                startIdx = mask;
                minDigit = curDigit-1;
            }
            else{
                minDigit = curDigit;
            }
        }
        if(startIdx>=0){
            num[startIdx]--;
            for(int i=startIdx+1;i<num.size();i++){
                num[i]='9';
            }
        }
        return stoi(num);
    }

    double minmaxGasDist(vector<int>& stations, int k) {
        double sm=0,bg;
        bg= stations[1]-stations[0];
        for(int i=2;i<stations.size();++i){
            double cur = stations[i]-stations[i-1];
            bg = max(bg,cur);
        }
        double error = 0.000001;
        while (sm+error<bg){
            double mid = (sm+bg)/2;
            int count = MGDHelper(stations,mid);
            if (count>k){
                sm=mid;
            }
            else{
                bg = mid;
            }
        }
        return sm;
    }

    int MGDHelper(const vector<int>& st,double gap){
        int count=0;
        for(int i=1;i<st.size();++i){
            double dist = st[i]-st[i-1];
            int res = dist/gap;
            count+= res+(gap*res<dist)-1;
        }
        return count;
    }
};

//int main() {
////    Solution s;
//    return 0;
//}