//Template class for a cell

template <typename T> class Cell {
    public:
        void update(T* neighbor) {
            static_cast<T*>(this)->update(neighbor);
        }
};
