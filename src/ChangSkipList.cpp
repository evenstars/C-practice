//
// Created by Heisenberg Liu on 2021/5/4.
//

#include <iostream>
#include "ChangSkipList.h"

namespace changSkipList {
    template<typename T>
    ChangSkipList<T>::ChangSkipList(int maxHeight, double nextLevelPossibility):maxHeight_(maxHeight),
                                                                                nextLevelPossibility_(
                                                                                        nextLevelPossibility) {
        assert(maxHeight > 0 && "max height should be positive");
        assert(nextLevelPossibility > 0 && nextLevelPossibility < 1 && "nextLevelPossibility should be between 0 - 1");
        headPtr_ = std::make_shared<Node<T>>(-1,-1, maxHeight_);
    }

    template<typename T>
    std::shared_ptr<Node<T>> ChangSkipList<T>::find(int key) {
        auto curNodePtr = headPtr_;
        std::shared_ptr<Node<T>> nextPtr;
        for (int curHeight = curMaxHeight_ - 1; curHeight >= 0; curHeight--) {
            while ((nextPtr = curNodePtr->next_[curHeight]) != nullptr && nextPtr->key_ < key) {
                curNodePtr = nextPtr;
            }
            if (nextPtr!= nullptr && nextPtr->key_ == key) {
                return nextPtr;
            }
        }
        return nullptr;
    }

    template<typename T>
    bool ChangSkipList<T>::insert(int key, T value) {
        std::vector<std::shared_ptr<Node<T>>> update(maxHeight_, headPtr_);
        auto curNodePtr = headPtr_;
        std::shared_ptr<Node<T>> nextPtr;
        for (int curHeight = curMaxHeight_ - 1; curHeight >= 0; curHeight--) {
            while ((nextPtr = curNodePtr->next_[curHeight]) != nullptr && nextPtr->key_ < key) {
                curNodePtr = nextPtr;
            }
            if (nextPtr != nullptr && nextPtr->key_ == key) {
                //for update
                nextPtr->value_ = value;
                return true;
            }
            update[curHeight]=curNodePtr;
        }
        int newNodeHeight = getHeight();
        curMaxHeight_ = std::max(curMaxHeight_, newNodeHeight);
        auto newNode = std::make_shared<Node<T>>(key, value, newNodeHeight);
        for (int i = 0; i < newNodeHeight; ++i) {
            newNode->next_[i] = update[i]->next_[i];
            update[i]->next_[i] = newNode;
        }
        return true;
    }

    template<typename T>
    bool ChangSkipList<T>::remove(int key) {
        std::vector<std::shared_ptr<Node<T>>> update(maxHeight_);
        auto curNodePtr = headPtr_;
        std::shared_ptr<Node<T>> nextPtr;
        for (int curHeight = curMaxHeight_ - 1; curHeight >= 0; curHeight--) {
            while ((nextPtr = curNodePtr->next_[curHeight]) != nullptr && nextPtr->key_ < key) {
                curNodePtr = nextPtr;
            }
            update[curHeight] = curNodePtr;
        }
        auto targetPtr = curNodePtr->next_[0];
        if (targetPtr == nullptr || targetPtr->key_ > key) {
            //not found
            return false;
        }
        for (int i = 0; i < targetPtr->height_; ++i) {
            update[i]->next_[i] = targetPtr->next_[i];
        }
        while (curMaxHeight_ && headPtr_->next_[curMaxHeight_ - 1] == nullptr) {
            curMaxHeight_--;
        }
        return true;
    }

    template<typename T>
    int ChangSkipList<T>::getHeight() {
        int curHeight = 1;
        for (; curHeight < maxHeight_ ; curHeight++){
            double r = ((double) rand() / (double) INT_MAX);
            if(r>nextLevelPossibility_){
                break;
            }
        }
        return curHeight;
    }

    template<typename T>
    void ChangSkipList<T>::display() {
        std::cout<<"\n*****Skip List*****\n"<<std::endl;
        for(int i=0;i<curMaxHeight_;++i){
            std::cout<<"level-"<<i<<": ";
            for(auto cur = headPtr_->next_[i];cur!= nullptr;cur=cur->next_[i]){
                std::cout<<cur->key_<<" ";
            }
            std::cout<<"\n";
        }
    }

}