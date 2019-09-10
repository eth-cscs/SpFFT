
program main
    use iso_c_binding
    use spfft
    implicit none
    integer :: i, j, k, counter
    integer, parameter :: dimX = 2
    integer, parameter :: dimY = 2
    integer, parameter :: dimZ = 2
    integer, parameter :: maxNumLocalZColumns = dimX * dimY
    integer, parameter :: processingUnit = 1
    integer, parameter :: maxNumThreads = -1
    type(c_ptr) :: grid = c_null_ptr
    type(c_ptr) :: transform = c_null_ptr
    integer :: errorCode = 0
    integer, dimension(dimX * dimY * dimZ * 3):: indices = 0
    complex(C_DOUBLE_COMPLEX), dimension(dimX * dimY * dimZ):: frequencyElements
    complex(C_DOUBLE_COMPLEX), pointer :: spaceDomain(:,:,:)
    type(c_ptr) :: realValuesPtr


    counter = 0
    do k = 1, dimZ
        do j = 1, dimY
            do i = 1, dimX
             frequencyElements(counter + 1) = cmplx(counter, -counter)
             indices(counter * 3 + 1) = i - 1
             indices(counter * 3 + 2) = j - 1
             indices(counter * 3 + 3) = k - 1
             counter = counter + 1
            end do
        end do
    end do

    ! print input
    print *, "Input:"
    do i = 1, size(frequencyElements)
         print *, frequencyElements(i)
    end do


    ! create grid and transform
    errorCode = spfft_grid_create(grid, dimX, dimY, dimZ, maxNumLocalZColumns, processingUnit, maxNumThreads);
    if (errorCode /= SPFFT_SUCCESS) error stop
    errorCode = spfft_transform_create(transform, grid, processingUnit, 0, dimX, dimY, dimZ, dimZ, size(frequencyElements), 0, indices)
    if (errorCode /= SPFFT_SUCCESS) error stop

    ! grid can be safely destroyed after creating all required transforms
    errorCode = spfft_grid_destroy(grid)
    if (errorCode /= SPFFT_SUCCESS) error stop

    ! set space domain array to use memory allocted by the library
    errorCode = spfft_transform_get_space_domain(transform, processingUnit, realValuesPtr)
    if (errorCode /= SPFFT_SUCCESS) error stop

    ! transform backward
    errorCode = spfft_transform_backward(transform, frequencyElements, processingUnit)
    if (errorCode /= SPFFT_SUCCESS) error stop


    call c_f_pointer(realValuesPtr, spaceDomain, [dimX,dimY,dimZ])

    print *, ""
    print *, "After backward transform:"
    do k = 1, size(spaceDomain, 3)
        do j = 1, size(spaceDomain, 2)
            do i = 1, size(spaceDomain, 1)
             print *, spaceDomain(i, j, k)
            end do
        end do
    end do

    ! transform forward (will invalidate space domain data)
    errorCode = spfft_transform_forward(transform, processingUnit, frequencyElements, 0)
    if (errorCode /= SPFFT_SUCCESS) error stop

    print *, ""
    print *, "After forward transform (without scaling):"
    do i = 1, size(frequencyElements)
             print *, frequencyElements(i)
    end do

    ! destroying the final transform will free the associated memory
    errorCode = spfft_transform_destroy(transform)
    if (errorCode /= SPFFT_SUCCESS) error stop

end
