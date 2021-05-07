//
// Created by Heisenberg Liu on 2021/5/4.
//

#ifndef PRACTICE_CHANGSKIPLIST_H
#define PRACTICE_CHANGSKIPLIST_H

#include <vector>
// A skip list supporting template, intelligent pointer

namespace changSkipList{

    template<typename T>
    struct Node{
        int key_;
        int height_;
        T value_;
        std::vector<std::shared_ptr<Node<T>>> next_;

        Node(int key,T value,int height):key_(key),value_(value),height_(height){
            next_ = std::vector<std::shared_ptr<Node<T>>>(height);
        }
    };

    template<typename T>
    class ChangSkipList {
    public:
        ChangSkipList()=default;
        ChangSkipList(int maxHeight,double nextLevelPossibility);
        ~ChangSkipList() = default;

        bool insert(int key, T value);
        std::shared_ptr<Node<T>> find(int key);
        bool remove(int key);
        void display();
    private:
        int maxHeight_=10;
        int curMaxHeight_=0;
        double nextLevelPossibility_ = 0.7;

        std::shared_ptr<Node<T>> headPtr_;

        int getHeight();
    };
}

int main(){
    using namespace changSkipList;
    ChangSkipList<int> sl(10,0.6);
    sl.insert(1,1);
    sl.insert(2,2);
    sl.display();
    sl.insert(3,3);
    sl.insert(4,4);
    sl.display();
    auto target = sl.find(3);

    sl.remove(2);
    sl.remove(3);
    sl.display();
    target = sl.find(3);
}

#endif //PRACTICE_CHANGSKIPLIST_H
