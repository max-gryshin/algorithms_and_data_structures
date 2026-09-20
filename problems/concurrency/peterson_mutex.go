package concurrency

import "go.uber.org/atomic"

type PetersonMutex struct {
	want    [2]atomic.Bool
	waiting atomic.Int64
}

func (pm *PetersonMutex) Lock(threadId int64) {
	other := 1 - threadId
	pm.want[threadId].Store(true)
	pm.waiting.Store(threadId)
	for pm.want[other].Load() &&
		pm.waiting.Load() == threadId {
	}
}

func (pm *PetersonMutex) Unlock(threadId int) {
	pm.want[threadId].Store(false)
}

func NewPetersonMutex() *PetersonMutex {
	return &PetersonMutex{}
}
