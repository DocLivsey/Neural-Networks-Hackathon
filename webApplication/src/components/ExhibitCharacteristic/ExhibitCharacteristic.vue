<script>
import {defineComponent} from 'vue'

export default defineComponent({
    name: "ExhibitCharacteristic",
    props: {
        characteristic: String,
        characteristicValue: String,
    }
})
</script>

<template>
    <div class="characteristic">
        <div class="characteristic__name">
            {{characteristic}}
        </div>
        <div class="characteristic__value">
            {{characteristicValue}}
        </div>
    </div>
</template>

<style scoped>
.characteristic {
    text-align: start;
    row-gap: 0.375em;
}
.characteristic__name {
    font-size: var(--secondary-font-size);
    font-family: var(--secondary-font-family);
    font-weight: 600;
}
.characteristic__value {
    font-size: var(--tertiary-font-size);
    font-family: var(--tertiary-font-family);
}
</style>